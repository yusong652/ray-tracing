import numpy as np

# print(np.inf > 1)
# arr = np.ones((4, 5, 3))
# arr[:, :] = None
# print(arr)
# print(len(arr.copy()))
class Sphere(object):
	num_sph = 0
	def __init__(self, 
		radius=1.0, 
		pos=np.array([0.0, 0.0, 0.0]),
		color=np.array([255.0, 255.0, 255.0])):
		self.id_sph = Sphere.num_sph
		Sphere.num_sph += 1

# sph1 = Sphere()
# print(sph1.id_sph)
# print(Sphere.num_sph)

# arr = np.ones((3, 3, 3))
# arr[2, 2, :] = np.array([1,1,1])

# arr = np.array([1, 2, 3])

# mat = np.array([
# 	[1, 0, 0],
# 	[0, 2, 0],
# 	[0, 0, 1]])

# arr_res = np.matmul(arr, mat)

# print(arr_res)
print((1+2+3))