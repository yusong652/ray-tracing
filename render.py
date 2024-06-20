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
        self.canvas_to_gui = ti.field(dtype=flt_default, shape=(self.camera.resolution[1], self.camera.resolution[0],3))
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
            color_local = self.intersect_ray_sphere(i, j, vec_d)
            self.num_pixel_rendered[0] += 1
            self.canvas[i, j, 0] = color_local[0]
            self.canvas[i, j, 1] = color_local[1]
            self.canvas[i, j, 2] = color_local[2]
            self.canvas_to_gui[j, i, 0] = color_local[0]
            self.canvas_to_gui[j, i, 1] = color_local[1]
            self.canvas_to_gui[j, i, 2] = color_local[2]
            # print("{} / {} pixels rendered".format(self.num_pixel_rendered[0], self.num_pixel_render[0]) )

    @ti.func
    def intersect_ray_sphere(self, i: ti.int32, j: ti.int32, vec_d: vec):
        color_local = vec(0, 0, 0)
        for index_particle in range(self.sphere.number):
            pos_sphere = vec(self.sphere.pos[index_particle, 0],
                             self.sphere.pos[index_particle, 1],
                             self.sphere.pos[index_particle, 2])
            vec_centroid_origin = self.camera.origin - pos_sphere
            t1, t2 = solve_quadratic_equation(vec_d, vec_centroid_origin, self.sphere.rad[index_particle])
            t = ti.math.min(t1, t2)
            if t < self.distance_object_close[i, j]:
                if t > self.render_lmt[0] and t < self.render_lmt[1]:

                    self.distance_object_close[i, j] = t
                    pos = vec_d * t + self.camera.origin
                    color_local = vec(self.sphere.color[index_particle, 0],
                                      self.sphere.color[index_particle, 1],
                                      self.sphere.color[index_particle, 2])
                    light_intensity = self.light_computer.compute_intensity(
                        self.sphere, index_particle, pos, self.camera)
                    color_local = clip(color_local * light_intensity, 0.0, 1.0)
            else:
                y = vec_d[1] / vec_d.norm()
                a = 0.5 * (y + 1.0)
                color_local = vec(1, 1, 1) * (1.0 - a) + vec(0.5, 0.7, 1.0) * a
        return color_local

    @ti.kernel
    def update_canvas_to_gui(self):
        for i, j in self.pixels:
            self.canvas_to_gui[j, i, 0] = self.canvas[i, j, 0]
            self.canvas_to_gui[j, i, 1] = self.canvas[i, j, 1]
            self.canvas_to_gui[j, i, 2] = self.canvas[i, j, 2]