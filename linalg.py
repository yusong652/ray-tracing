import taichi as ti

vec = ti.math.vec3


def euler_rot(yaw: ti.f32, pitch: ti.f32, roll: ti.f32) -> ti.math.mat3:
    res = ti.math.mat3([
        [ti.math.cos(yaw) * ti.math.cos(pitch),
         ti.math.cos(yaw) * ti.math.sin(pitch) * ti.math.sin(roll) -
         ti.math.sin(yaw) * ti.math.cos(roll),
         ti.math.cos(yaw) * ti.math.sin(pitch) * ti.math.cos(roll) +
         ti.math.sin(yaw) * ti.math.sin(roll)],
        [ti.math.sin(yaw) * ti.math.cos(pitch),
         ti.math.sin(yaw) * ti.math.sin(pitch) * ti.math.sin(roll) +
         ti.math.cos(yaw) * ti.math.cos(roll),
         ti.math.sin(yaw) * ti.math.sin(pitch) * ti.math.cos(roll) -
         ti.math.cos(yaw) * ti.math.sin(roll)],
        [-ti.math.sin(pitch),
         ti.math.cos(pitch) * ti.math.sin(roll),
         ti.math.cos(pitch) * ti.math.cos(roll)]])

    return res

