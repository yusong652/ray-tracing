import taichi as ti
from linalg import euler_rot
from format import flt_default, INF
vec = ti.math.vec3


@ti.data_oriented
class Camera(object):
    def __init__(self):
        self.origin = vec(-1.35, -0.29, 0.0)
        self.distance = 1.0
        self.resolution = ti.field(dtype=ti.i32, shape=(2,))
        self.resolution[0] = 1080
        self.resolution[1] = 1920
        self.height = 0.5
        self.width = self.height * self.resolution[1]/self.resolution[0]
        self.yaw = ti.math.pi * 0.0
        self.pitch = ti.math.pi * -0.5
        self.roll = ti.math.pi * -0.0
        self.rotation = euler_rot(self.yaw, self.pitch, self.roll)
        # vectors from origin to viewport
        self.vec_o_to_vp = ti.field(dtype=flt_default, shape=(self.resolution[0], self.resolution[1], 3))
        self.calculate_o_to_vp()

    @ti.kernel
    def calculate_o_to_vp(self):
        for i, j in ti.ndrange(self.resolution[0], self.resolution[1]):
            vec_d = ti.math.vec3((j - self.resolution[1] / 2.0) / self.resolution[1] * self.width,
                        (i - self.resolution[0] / 2.0) / self.resolution[0] * self.height, self.distance)
            vec_d = vec_d @ self.rotation

            self.vec_o_to_vp[i, j, 0] = vec_d[0]
            self.vec_o_to_vp[i, j, 1] = vec_d[1]
            self.vec_o_to_vp[i, j, 2] = vec_d[2]
