import taichi as ti
from format import INF, flt_default
from linalg import solve_quadratic_equation
vec = ti.math.vec3


@ti.data_oriented
class LightComputer(object):
    def __init__(self):
        self.intensity_amb = 0.42  # ambient light
        self.intensity_dir = 0.36  # directional light
        self.intensity_point = 0.38  # point light
        self.light_dir = vec(-1.0, 0.5, 0.5).normalized()  # point to light source
        self.pos_light = vec(-2.0, 3.0, 2.0)  # point light

    @ti.func
    def compute_intensity(self, sphere: ti.template(), index_p: ti.int32, pos: vec, camera: ti.template()):
        intensity = 0.0
        intensity += self.intensity_amb
        intensity += self.compute_diffuse(sphere, index_p, pos)
        intensity += self.compute_spec(sphere, index_p, pos, camera)

        return intensity

    @ti.func
    def check_shadow(self, pos: vec, sphere: ti.template(), state: ti.int32) -> flt_default:
        """
        check if the shadow is cast under the directional light
        state 0 means directional light
        state 1 means point light
        """
        # shadow coefficient 1.0 means the point is not shadowed by other objects
        # shadow coefficient 0.0 means the point is shadowed by other objects
        shadow_coefficient = 1.0
        vec_d = self.light_dir
        lmt_min = 1.0e-3
        lmt_max = INF
        if state == 0:
            vec_d = self.light_dir
            # Minimum limitation is set to avoid self-cast
            lmt_min = 1.0e-3
            lmt_max = INF
        else:
            vec_d = self.pos_light - pos
            lmt_min = 1.0e-3
            lmt_max = 1.0
        for index_p in range(sphere.number):
            # vector from the centroid of sphere to the position

            vec_co = pos - vec(sphere.pos[index_p, 0], sphere.pos[index_p, 1], sphere.pos[index_p, 2])
            radius = sphere.rad[index_p]
            t1, t2 = solve_quadratic_equation(vec_d, vec_co, radius)

            if lmt_min < t1 < lmt_max or lmt_min < t2 < lmt_max:
                shadow_coefficient = 0.0
                break
            else:
                pass
        return shadow_coefficient

    @ti.func
    def compute_spec(self, sphere: ti.template(), index_p: ti.int32, pos: vec, camera: ti.template()) -> flt_default:
        """
        specular reflection
        """
        intensity = 0.0
        if sphere.specular[index_p] != -1:  # -1 refers to matte object
            # directional
            # vector from the centroid to the point
            vec_norm = pos - vec(sphere.pos[index_p, 0], sphere.pos[index_p, 1], sphere.pos[index_p, 2])
            vec_norm = vec_norm.normalized()
            # reflection vector
            vec_r_dir = vec_norm * self.light_dir.dot(vec_norm) * 2 - self.light_dir
            # view vector
            origin = camera.origin
            vec_v = origin - pos
            vec_v = vec_v.normalized()
            # directional light reflection vector
            prod_vr_dir = vec_r_dir.dot(vec_v)
            if prod_vr_dir > 0:
                shadow_coefficient = self.check_shadow(pos, sphere, 0)
                intensity_dir = (self.intensity_dir * prod_vr_dir ** sphere.specular[index_p] *
                                 shadow_coefficient)
                intensity += intensity_dir

            else:
                pass

            # point light
            vec_pl = self.pos_light - pos  # pointing to source
            vec_pl = vec_pl.normalized()
            # point light reflection vector
            vec_r_point = vec_norm * vec_pl.dot(vec_norm) * 2 - vec_pl
            prod_vr_point = vec_r_point.dot(vec_v)
            if prod_vr_point > 0:
                shadow_coefficient = self.check_shadow(pos, sphere, 1)
                intensity_point = (self.intensity_point * prod_vr_point ** sphere.specular[index_p] *
                                   shadow_coefficient)
                intensity += intensity_point
            else:
                pass
        else:
            pass
        return intensity



    @ti.func
    def compute_diffuse(self, sphere: ti.template(), index_p: ti.int32, pos: vec) -> flt_default:
        intensity = 0.0
        # vector from the centroid to the point
        vec_norm = pos - vec(sphere.pos[index_p, 0], sphere.pos[index_p, 1], sphere.pos[index_p, 2])
        vec_norm = vec_norm.normalized()
        # inner product of directional light (pointing to source) and radius vector
        prod_dir = vec_norm.dot(self.light_dir)
        if prod_dir >= 0.0:
            shadow_coefficient = self.check_shadow(pos, sphere, state=0)
            intensity_dir = prod_dir * self.intensity_dir * shadow_coefficient
            intensity += intensity_dir
        else:
            pass

        vec_pl = self.pos_light - pos
        vec_pl = vec_pl.normalized()
        prod_point = vec_norm.dot(vec_pl)
        # point light
        if prod_point >= 0.0:
            shadow_coefficient = self.check_shadow(pos, sphere, state=1)
            intensity_point = prod_point * self.intensity_point * shadow_coefficient
            intensity += intensity_point
        else:
            pass

        return intensity
