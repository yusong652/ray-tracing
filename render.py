import taichi as ti
from camera import Camera
from sphere import Sphere
from format import flt_default, INF
from light_comput import LightComputer
from linalg import solve_quadratic_equation, clip

vec = ti.math.vec3

@ti.data_oriented
class Renderer:
    def __init__(self):
        self.camera = Camera()
        self.pixels = ti.field(dtype=flt_default, shape=(self.camera.resolution[0], self.camera.resolution[1]))
        self.pixels_rendered = ti.field(dtype=ti.i32, shape=(self.camera.resolution[0], self.camera.resolution[1]))
        self.num_pixel_render = ti.field(dtype=ti.i32, shape=(1,))
        self.num_pixel_render[0] = self.camera.resolution[0] * self.camera.resolution[1]
        self.num_pixel_rendered = ti.field(dtype=ti.i32, shape=(1,))
        self.canvas = ti.field(dtype=flt_default, shape=(self.camera.resolution[0], self.camera.resolution[1], 3))
        self.canvas_to_gui = ti.field(dtype=ti.f32,
                                      shape=(self.camera.resolution[1], self.camera.resolution[0], 3))
        self.canvas.fill(1.0)
        self.sphere = Sphere()
        # record the distance of the closest object, initiated as infinite
        self.distance_object_close = ti.field(
            dtype=flt_default, shape=(self.camera.resolution[0], self.camera.resolution[1]))
        self.distance_object_close.fill(INF)
        self.render_lmt = ti.field(dtype=flt_default, shape=(2,))
        self.render_lmt[0] = self.camera.distance
        self.render_lmt[1] = INF
        self.light_computer = LightComputer()

    @ti.kernel
    def render(self, supersample: ti.i32):
        for i, j in self.pixels:
            if self.pixels_rendered[i, j] == 1:
                continue
            isRandomSample = 1
            if isRandomSample:
                if ti.random() > 0.01:
                    continue
            color_accum = vec(0.0, 0.0, 0.0)
            for sample in range(supersample):
                u_offset = ti.random() / self.camera.resolution[0] * 1.0
                v_offset = ti.random() / self.camera.resolution[1] * 1.0
                vec_d = vec(self.camera.vec_o_to_vp[i, j, 0] + u_offset,
                        self.camera.vec_o_to_vp[i, j, 1] + v_offset,
                        self.camera.vec_o_to_vp[i, j, 2])
                color = clip(self.trace_color(self.camera.origin, vec_d, 1, 4),
                            0.0, 1.0)
                color_accum += color
            color_avg = color_accum / supersample
            self.num_pixel_rendered[0] += 1
            self.canvas[i, j, 0] = color_avg[0]
            self.canvas[i, j, 1] = color_avg[1]
            self.canvas[i, j, 2] = color_avg[2]
            self.canvas_to_gui[j, i, 0] = color_avg[0]
            self.canvas_to_gui[j, i, 1] = color_avg[1]
            self.canvas_to_gui[j, i, 2] = color_avg[2]
            self.pixels_rendered[i, j] = 1
            # print("{} / {} pixels rendered".format(self.num_pixel_rendered[0], self.num_pixel_render[0]))

    @ti.func
    def get_bg_color(self, vec_d: vec) -> vec:
        y = vec_d[1] / vec_d.norm()
        a = 0.5 * (y + 1.0)
        color_bg = vec(1, 1, 1) * (1.0 - a) + vec(0.3, 0.4, 0.8) * a
        return color_bg

    @ti.func
    def get_reflect_ray(self, vec_d: vec, vec_n: vec) -> vec:
        return vec_d - 2 * vec_d.dot(vec_n) * vec_n
    
    @ti.func
    def refract(self, vec_d: vec, vec_n: vec, eta: flt_default) -> vec:
        """
        计算折射光线方向
        :param vec_d: 入射光线方向
        :param vec_n: 法线方向
        :param eta: 折射率
        :return: 折射光线方向
        """
        refracted = vec(0.0, 0.0, 0.0)  # 默认零向量
        cos_theta_i = -vec_n.dot(vec_d)
        sin2_theta_t = eta * eta * (1.0 - cos_theta_i * cos_theta_i)
        
        # 如果没有全内反射，计算折射光线
        if sin2_theta_t <= 1.0:
            cos_theta_t = ti.sqrt(1.0 - sin2_theta_t)
            refracted = eta * vec_d + (eta * cos_theta_i - cos_theta_t) * vec_n
        
        return refracted.normalized()
    
    @ti.func
    def fresnel_schlick(self, cos_theta: float, refractive_index: float) -> float:
        r0 = ((1.0 - refractive_index) / (1.0 + refractive_index)) ** 2
        return r0 + (1.0 - r0) * ((1.0 - cos_theta) ** 5)

    @ti.func
    def trace_color(self, origin, vec_d: vec, t_min: flt_default, recursion_depth: ti.template()) -> vec:
        t_closest = self.render_lmt[1]
        color = self.get_bg_color(vec_d)
        for index_particle in range(self.sphere.number):
            pos_sphere = self.sphere.get_pos(index_particle)
            vec_centroid_origin = origin - pos_sphere
            t1, t2 = solve_quadratic_equation(vec_d, vec_centroid_origin, self.sphere.rad[index_particle])
            t = ti.math.min(t1, t2)
            if t_min < t < t_closest:
                t_closest = t
                pos = origin + t * vec_d
                vec_n = (pos - pos_sphere).normalized()

                # 本地颜色
                color_local = self.sphere.get_color(index_particle, pos) * self.light_computer.compute_intensity(
                    self.sphere, index_particle, pos, self.camera)

                reflect_ratio = self.sphere.reflective[index_particle]
                refract_index = self.sphere.refraction_index[index_particle]
                refract_ratio = self.sphere.refractive[index_particle]
                reflected_color = vec(0.0, 0.0, 0.0)
                refracted_color = vec(0.0, 0.0, 0.0)

                if ti.static(recursion_depth > 0):
                    if refract_ratio > 0.0:
                        cos_theta = -vec_d.dot(vec_n)
                        eta = 1.0 / refract_index if vec_d.dot(vec_n) < 0 else refract_index
                        vec_n = vec_n if vec_d.dot(vec_n) < 0 else -vec_n
                        refracted_vec = self.refract(vec_d, vec_n, eta)
                        if refracted_vec.norm() > 0.0:
                            refracted_color = self.trace_color(pos, refracted_vec, 0.001, recursion_depth - 1) * refract_ratio
                            refracted_color = refracted_color * refract_ratio

                    # 反射光
                    vec_reflect = self.get_reflect_ray(vec_d, vec_n)
                    reflected_color = self.trace_color(pos, vec_reflect, 0.001, recursion_depth - 1) * reflect_ratio

                # 混合颜色
                local_weight = max(0.0, 1.0 - reflect_ratio - refract_ratio)
                color = color_local * local_weight + reflected_color + refracted_color
        return color

    @ti.kernel
    def update_canvas_to_gui(self):
        for i, j in self.pixels:
            self.canvas_to_gui[j, i, 0] = self.canvas[i, j, 0]
            self.canvas_to_gui[j, i, 1] = self.canvas[i, j, 1]
            self.canvas_to_gui[j, i, 2] = self.canvas[i, j, 2]
