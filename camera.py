import taichi as ti
from matrix_trans import
vec = ti.math.vec3



class Camera(object):
    def __init__(self):
        self.origin = vec(0.0, 0.0, 0.0)
        self.distance = 1.0
        self.resolution = ti.field(dtype=ti.i32, shape=(2,))
        self.resolution[0] = 1920
        self.resolution[1] = 1080
        self.height = 1.0
        self.width = self.height * self.resolution[0]/self.resolution[1]
        self.yaw = ti.math.pi * 0.0
        self.pitch = ti.math.pi * 0.0
        self.roll = ti.math.pi * 0.08
        self.rotation = ti.