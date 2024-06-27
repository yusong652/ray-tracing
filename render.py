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
    def render(self):
        for i, j in self.pixels:
            vec_d = vec(self.camera.vec_o_to_vp[i, j, 0],
                        self.camera.vec_o_to_vp[i, j, 1],
                        self.camera.vec_o_to_vp[i, j, 2])
            color = clip(self.trace_color(self.camera.origin, vec_d, 1, 2),
                         0.0, 1.0)
            self.num_pixel_rendered[0] += 1
            self.canvas[i, j, 0] = color[0]
            self.canvas[i, j, 1] = color[1]
            self.canvas[i, j, 2] = color[2]
            self.canvas_to_gui[j, i, 0] = color[0]
            self.canvas_to_gui[j, i, 1] = color[1]
            self.canvas_to_gui[j, i, 2] = color[2]
            print("{} / {} pixels rendered".format(self.num_pixel_rendered[0], self.num_pixel_render[0]))

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
    def trace_color(self, origin, vec_d: vec, t_min: flt_default, recursion_depth: ti.template()) -> flt_default:
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
                color_local = self.sphere.get_color(index_particle) * self.light_computer.compute_intensity(
                    self.sphere, index_particle, pos, self.camera)
                if ti.static(recursion_depth <= 0):
                    color = color_local
                else:
                    vec_n = (pos - pos_sphere).normalized()
                    vec_r = self.get_reflect_ray(vec_d, vec_n)
                    r = self.sphere.reflective[index_particle]
                    color = color_local * (1.0 - r) + self.trace_color(
                        pos, vec_r, 1.0e-3, recursion_depth - 1) * r
        return color

    @ti.kernel
    def update_canvas_to_gui(self):
        for i, j in self.pixels:
            self.canvas_to_gui[j, i, 0] = self.canvas[i, j, 0]
            self.canvas_to_gui[j, i, 1] = self.canvas[i, j, 1]
            self.canvas_to_gui[j, i, 2] = self.canvas[i, j, 2]
