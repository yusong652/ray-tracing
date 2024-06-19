import numpy as np
import matplotlib.pyplot as plt

pixel = (540, 960)
canvas = np.ones((pixel[0], pixel[1], 3)) * 255.0
sph_close = np.ones((pixel[0], pixel[1]))
sph_close[:, :] = None
origin = (0.0, 0.0, 0.0)


class Sphere(object):
	num_sph = 0
	def __init__(self, 
		radius=1.0, 
		pos=np.array([0.0, 0.0, 0.0]),
		color=np.array([255.0, 255.0, 255.0]),
		spec=16.0
		):
		self.radius = radius
		self.pos = (pos)
		self.color = color
		self.id_sph = Sphere.num_sph
		self.spec = spec
		Sphere.num_sph += 1


sphs = []
sph1 = Sphere(radius=0.36, pos=np.array([0.4, 0.5, 1.5]),
	color = np.array([222.0, 46.0, 56.0]))
sph1.spec = 512.0
sphs.append(sph1)
sph2 = Sphere(radius=0.38, pos=np.array([-0.38, 0.15, 1.8]),
	color = np.array([0.0, 166.0, 166.0]))
sphs.append(sph2)


class Camera(object):
	def __init__(self, pixel=pixel):
		self.origin = np.array([0.0, 0.0, 0.0])
		self.distance = 1.0
		self.pixel = pixel
		self.width = self.pixel[1]/self.pixel[0]
		self.height = 1.0
		self.yaw = 0.0 * np.pi
		self.pitch = 0.0 * np.pi
		self.roll = 0.08 * np.pi
		self.rot_mat = np.array([
			[np.cos(self.yaw)*np.cos(self.pitch),
			np.cos(self.yaw)*np.sin(self.pitch)*np.sin(self.roll)-
			np.sin(self.yaw)*np.cos(self.roll),
			np.cos(self.yaw)*np.sin(self.pitch)*np.cos(self.roll)+
			np.sin(self.yaw)*np.sin(self.roll)], 
			[np.sin(self.yaw)*np.cos(self.pitch), 
			np.sin(self.yaw)*np.sin(self.pitch)*np.sin(self.roll)+
			np.cos(self.yaw)*np.cos(self.roll),
			np.sin(self.yaw)*np.sin(self.pitch)*np.cos(self.roll)-
			np.cos(self.yaw)*np.sin(self.roll)], 
			[-np.sin(self.pitch), 
			np.cos(self.pitch)*np.sin(self.roll),
			np.cos(self.pitch)*np.cos(self.roll)]])


class LightComputer(object):
	def __init__(self):
		self.intensity_amb = 0.2 # ambient light
		self.intensity_dir = 0.3 # directional light
		self.intensity_point = 0.5 # point light 
		self.vec_dir = np.array([1, 1, 2.0]) # pointing to source
		self.vec_dir = self.vec_dir / np.sqrt(self.vec_dir[0]**2+
			self.vec_dir[1]**2+self.vec_dir[2]**2)
		self.pos_light = np.array([5.0, 4.0, -5.0]) # point light

	def compute_intensity(self, sph, pos, origin):
		intensity = 0.0
		intensity += self.intensity_amb

		intensity += self.compute_diffuse(sph, pos)
		intensity += self.compute_spec(sph, pos, origin)

		return intensity

	def check_shadow(self, pos, sphs, state='dir'):
		"""
		check if the shadow is casted under the directional light
		"""
		shadow_state = False
		if state == 'dir':
			vec_d = self.vec_dir
			lmt_min = 1e-7
			lmt_max = np.inf
		elif state == 'point':
			vec_d = self.pos_light - pos
			lmt_min = 1e-7
			lmt_max = 1.0
		for sph in sphs:
			# vector from the centroid of sphere to the position
			vec_co = pos - sph.pos 
			radius = sph.radius
			a = vec_d[0]**2 + vec_d[1]**2 + vec_d[2]**2
			b = (vec_co[0]*vec_d[0] +
				vec_co[1]*vec_d[1] +
				vec_co[2]*vec_d[2]) * 2.0
			c = vec_co[0]**2 + vec_co[1]**2 + vec_co[2]**2 - radius**2
			# discriminant
			disc = b**2 - 4*a*c
			if disc < 0.0:
				# no solution 
				continue
			else:
				t1 = (-b + np.sqrt(disc)) / (a*2.0)
				t2 = (-b - np.sqrt(disc)) / (a*2.0)
				if t1 > lmt_min and t1 < lmt_max:
					shadow_state = True
					break
				else:
					pass
				if t2 > lmt_min and t2 < lmt_max:
					shadow_state = True
					break
				else:
					pass
		return shadow_state

	def compute_diffuse(self, sph, pos):
		intensity = 0.0
		# ambient light
		intensity += self.intensity_amb
		vec_norm = pos - sph.pos # vector from the centroid to the point
		vec_norm = vec_norm / np.sqrt(vec_norm[0]**2+
			vec_norm[1]**2+
			vec_norm[2]**2)

		# directional light
		prod_dir = (vec_norm[0]*self.vec_dir[0]+
			vec_norm[1]*self.vec_dir[1]+
			vec_norm[2]*self.vec_dir[2])
		if  prod_dir >= 0:
			if not self.check_shadow(pos, sphs, state='dir'):
				intensity += prod_dir * self.intensity_dir
			else:
				pass
		else:
			pass
		vec_pl = self.pos_light - pos
		vec_pl = vec_pl / np.sqrt(vec_pl[0]**2+
			vec_pl[1]**2+
			vec_pl[2]**2)
		prod_point = (vec_norm[0]*vec_pl[0]+
			vec_norm[1]*vec_pl[1]+
			vec_norm[2]*vec_pl[2]) 
		# point light
		if prod_point >= 0:
			if not self.check_shadow(pos, sphs, state='point'):
				intensity += prod_point * self.intensity_point
			else:
				pass
		else:
			pass

		return intensity

	def compute_spec(self, sph, pos, origin):
		"""
		specular reflecti
		"""
		intensity = 0.0
		if sph.spec != -1: # -1 refers to matte object
			# directional
			# vector from the centroid to the point
			vec_norm = pos - sph.pos 
			vec_norm = vec_norm / np.sqrt(vec_norm[0]**2+
				vec_norm[1]**2+
				vec_norm[2]**2)
			# reflection vector
			vec_r_dir = (
				vec_norm*(self.vec_dir[0]*vec_norm[0]+
					self.vec_dir[1]*vec_norm[1]+
					self.vec_dir[2]*vec_norm[2])*2-
				self.vec_dir)
			# view vecor
			vec_v = origin - pos
			vec_v = vec_v/np.sqrt(vec_v[0]**2+
				vec_v[1]**2+
				vec_v[2]**2)
			# directional light reflection vector
			prod_vr_dir = (vec_r_dir[0]*vec_v[0]+
				vec_r_dir[1]*vec_v[1]+
				vec_r_dir[2]*vec_v[2])
			if prod_vr_dir > 0:
				if not self.check_shadow(pos, sphs, state='dir'):
					intensity += (
						self.intensity_dir * prod_vr_dir**sph.spec)
				else:
					pass
			else:
				pass
			# point light
			vec_pl = self.pos_light - pos # pointing to source
			vec_pl = vec_pl / np.sqrt(vec_pl[0]**2+
				vec_pl[1]**2+
				vec_pl[2]**2)
			# point light reflection vector
			vec_r_point = (
				vec_norm*(vec_pl[0]*vec_norm[0]+
					vec_pl[1]*vec_norm[1]+
					vec_pl[2]*vec_norm[2])*2-
				vec_pl)
			prod_vr_point = (
				vec_r_point[0]*vec_v[0]+
				vec_r_point[1]*vec_v[1]+
				vec_r_point[2]*vec_v[2])
			if prod_vr_point > 0:
				if not self.check_shadow(pos, sphs, state='point'):
					intensity += (
						self.intensity_point * prod_vr_point**sph.spec)
				else:
					pass
			else:
				pass
		else:
			pass
		return intensity


class Renderer(object):
	def __init__(self, pixel, canvas, sphs, sph_close):
		self.pixel = pixel
		self.origin = origin
		self.camera = Camera()
		self.canvas = canvas
		self.sphs = sphs
		self.dist_close = np.ones(self.pixel)*np.inf
		self.sph_close = sph_close
		self.render_lmt = np.array([1.0, 1.0e6])
		self.lc = LightComputer()

	def trace_ray(self, row, col):
		# vector from the origin to the pixel on the viewport
		vec_d = np.array([
			(col-self.pixel[1]/2.0)/self.pixel[1]*self.camera.width,
			(row-self.pixel[0]/2.0)/self.pixel[0]*self.camera.height,
			self.camera.distance])
		vec_d = self.rot_vec(vec_d, self.camera.rot_mat)

		# print(vec_d)
		for sph in self.sphs:
			self.intersectRaySphere(sph, vec_d, row, col)

	def rot_vec(self, vec_ori, rot_mat):
		return np.matmul(vec_ori, rot_mat)

	def intersectRaySphere(self, sph, vec_d, row, col):
		# vector from the centroid to origin
		vec_co = self.origin - sph.pos
		radius = sph.radius
		a = vec_d[0]**2 + vec_d[1]**2 + vec_d[2]**2
		b = (vec_co[0]*vec_d[0] +
			vec_co[1]*vec_d[1] +
			vec_co[2]*vec_d[2]) * 2.0
		c = vec_co[0]**2 + vec_co[1]**2 + vec_co[2]**2 - radius**2
		# discriminant
		disc = b**2 - 4*a*c
		if disc < 0.0:
			# no solution 
			self.dist_close[row, col] = np.inf
		else:
			t1 = (-b + np.sqrt(disc)) / (a*2.0)
			t2 = (-b - np.sqrt(disc)) / (a*2.0)
			if t1 > self.render_lmt[0] and t1 < self.render_lmt[1]:
				if t1 < self.dist_close[row, col]:
					self.dist_close[row, col] = t1
					pos = vec_d * t1
					light_intensity = self.lc.compute_intensity(
						sph, pos, self.camera.origin)
					self.canvas[row, col, :] = sph.color*light_intensity
				else:
					pass
			if t2 > self.render_lmt[0] and t2 < self.render_lmt[1]:
				if t2 <self.dist_close[row, col]:
					self.dist_close[row, col] = t2
					pos = vec_d * t2
					light_intensity = self.lc.compute_intensity(
						sph, pos, self.camera.origin)
					color_rgb = np.clip(
						sph.color*light_intensity, 0, 255.0)
					self.canvas[row, col, :] = color_rgb
				else:
					pass
			else:
				pass

	def render_pixel(self):
		for row in range(self.pixel[0]):
			for col in range(self.pixel[1]):
				self.trace_ray(row, col,)


renderer = Renderer(pixel, canvas, sphs, sph_close)
renderer.render_pixel()

fig = plt.figure(figsize=(pixel[1]/300, pixel[0]/300))
rgb_map = renderer.canvas.copy()
rgb_map[:,:,:] = rgb_map[:, :, :] / 255.0
plt.imshow(rgb_map)
plt.gca().invert_yaxis()
plt.gca().set_axis_off()
plt.savefig('shadow.png', dpi=500)
plt.show()
