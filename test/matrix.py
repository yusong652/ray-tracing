import taichi as ti
vec = ti.math.vec3

ti.init(arch=ti.cpu)
mat_1 = ti.Matrix([[0, 1, 2],
                   [2, 3, 2]])

mat_2 = ti.Matrix([[0, 0],
                   [0, 1],
                   [1, 0]])
array1 = ti.field(dtype=ti.f32, shape=(3, 3))
vec_1 = vec(0, 2, 3)
vec_2 = vec(2, 0, 3)

res = mat_1 @ mat_2
res_2 = vec_1 @ mat_2
# print(res)
# print(res_2)
#
# @ti.kernel
# def test():
#     for i, j in array1:
#         print(ti_func(vec_1, vec_2))
#
# @ti.func
# def ti_func(vec1: vec, vec2: vec) -> vec:
#     return vec1 + vec2

# test()

print(ti.math.max(vec_1, vec_2))