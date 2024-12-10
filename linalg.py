import taichi as ti
from format import flt_default, INF
vec = ti.math.vec3


def euler_rot(yaw: flt_default, pitch: flt_default, roll: flt_default) -> ti.math.mat3:
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

@ti.func
def solve_quadratic_equation(vec_1: vec, vec_2: vec, radius: flt_default) -> flt_default:
    a = vec_1[0] ** 2 + vec_1[1] ** 2 + vec_1[2] ** 2
    b = (vec_2[0] * vec_1[0] +
         vec_2[1] * vec_1[1] +
         vec_2[2] * vec_1[2]) * 2.0
    c = vec_2[0] ** 2 + vec_2[1] ** 2 + vec_2[2] ** 2 - radius ** 2
    # discriminant
    disc = b ** 2 - 4 * a * c
    res1, res2 = INF, INF
    if disc < 0.0:
        # no solution
        pass
    else:
        t1 = (-b + ti.sqrt(disc)) / (a * 2.0)
        t2 = (-b - ti.sqrt(disc)) / (a * 2.0)
        res1, res2 = t1, t2
    return res1, res2


@ti.func
def clip(val_ori: vec, lower_bound: flt_default, upper_bound: flt_default) -> vec:
    res = val_ori
    for i in range(len(res)):
        if res[i] > upper_bound:
            res[i] = upper_bound
        if res[i] < lower_bound:
            res[i] = lower_bound
    return res
