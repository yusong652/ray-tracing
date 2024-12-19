import numpy as np
import taichi as ti
import pandas as pd
from format import flt_default
from colormap import scalar_to_rgb
vec = ti.math.vec3


@ti.data_oriented
class Sphere(object):
    def __init__(self):

        self.number = None
        self.rad = None
        self.rad_max = None
        self.rad_min = None
        self.pos = None
        self.color = None
        self.specular = None
        self.reflective = None
        self.load_file()
        self.get_rad_range()
        self.set_rad_colormap()

    @ti.func
    def set_radius(self, i: ti.i32, rad: flt_default):
        self.rad[i] = rad

    @ti.func
    def set_pos(self, i: ti.i32, pos: vec):
        self.pos[i, 0] = pos[0]
        self.pos[i, 1] = pos[1]
        self.pos[i, 2] = pos[2]

    def set_color(self, i: ti.i32, color: vec):
        self.color[i, 0] = color[0]
        self.color[i, 1] = color[1]
        self.color[i, 2] = color[2]

    @ti.func
    def set_specular(self, i: ti.i32, specular: flt_default):
        self.specular[i] = specular

    def load_file(self):
        try:
            file_name = 'ball_info_0.csv'
            df = pd.read_csv(file_name)
            self.number = df.shape[0] - 50000 + 1
            self.rad = ti.field(dtype=flt_default, shape=(self.number,))
            self.pos = ti.field(dtype=flt_default, shape=(self.number, 3))
            self.color = ti.field(dtype=flt_default, shape=(self.number, 3))
            self.texture = ti.field(dtype=ti.i32, shape=(self.number))
            self.specular = ti.field(dtype=flt_default, shape=(self.number,))
            self.reflective = ti.field(dtype=flt_default, shape=(self.number))
            self.refractive = ti.field(dtype=flt_default, shape=(self.number))
            self.refraction_index = ti.field(dtype=flt_default, shape=(self.number))
            radii = df['rad'].to_numpy()
            pos_x = df['pos_x'].to_numpy()
            pos_y = df['pos_y'].to_numpy()
            pos_z = df['pos_z'].to_numpy()
            # specular = df['specular'].to_numpy()
            for i in range(self.number - 1):
                self.rad[i] = radii[i]
                self.pos[i, 0] = pos_x[i]
                self.pos[i, 1] = pos_y[i]
                self.pos[i, 2] = pos_z[i]
                self.specular[i] = 64
                self.reflective[i] = 0.4
                self.refractive[i] = 0.1
                self.refraction_index[i] = 1.5
            self.rad[self.number - 1] = 1024*6
            self.pos[self.number - 1, 0] = 1.5
            self.pos[self.number - 1, 1] = -1024*6 - 2.2
            self.pos[self.number - 1, 2] = 0.3
            self.specular[self.number - 1] = 512
            self.reflective[self.number - 1] = 0
            self.refractive[self.number - 1] = 0.0
            self.texture[self.number - 1] = 1

        except FileNotFoundError:
            self.default_init()

    def default_init(self):
        self.number = 3
        self.rad = ti.field(dtype=flt_default, shape=(self.number,))
        self.pos = ti.field(dtype=flt_default, shape=(self.number, 3))
        self.color = ti.field(dtype=flt_default, shape=(self.number, 3))
        self.specular = ti.field(dtype=flt_default, shape=(self.number,))
        self.reflective = ti.field(dtype=flt_default, shape=(self.number,))
        self.refractive = ti.field(dtype=flt_default, shape=(self.number))
        self.refraction_index = ti.field(dtype=flt_default, shape=(self.number))
        self.texture = ti.field(dtype=ti.i32, shape=(self.number))
        self.rad[0] = 0.28
        self.pos[0, 0] = 1.5
        self.pos[0, 1] = -0.3
        self.pos[0, 2] = -0.3
        self.specular[0] = 16.0
        self.reflective[0] = 0.4
        self.refractive[0] = 0.3
        self.refraction_index[0] = 1.5
        self.rad[1] = 0.26
        self.pos[1, 0] = 1.5
        self.pos[1, 1] = -0.3
        self.pos[1, 2] = 0.3
        self.specular[1] = 64
        self.reflective[1] = 0.1
        self.refractive[1] = 0.8
        self.refraction_index[1] = 1.5
        self.rad[2] = 1024*6
        self.pos[2, 0] = 1.5
        self.pos[2, 1] = -1024*6 - 2.2
        self.pos[2, 2] = 0.3
        self.specular[2] = 512
        self.reflective[2] = 0.4
        self.refractive[2] = 0.0
        self.texture[2] = 1

    @ti.func
    def get_pos(self, i: ti.i32):
        return vec(self.pos[i, 0], self.pos[i, 1], self.pos[i, 2])

    @ti.func
    def get_color(self, i: ti.i32, pos: vec):
        color = vec(self.color[i, 0], self.color[i, 1], self.color[i, 2])
        if self.texture[i] == 1:
            stripLen = 1.0e0
            vec_p = pos - self.get_pos(i)
            checker = (ti.floor(vec_p[0]/stripLen) + ti.floor(vec_p[2]/stripLen)) % 2
            if checker == 1:
                color = vec(0.21, 0.41, 0.35)
            else:
                color = vec(1.0, 0.98, 0.9)
        return color

    def get_rad_range(self):
        rad_max = self.rad[0]
        rad_min = self.rad[0]
        for i in range(self.number-1):
            if self.rad[i] >= rad_max:
                rad_max = self.rad[i]
            if self.rad[i] <= rad_min:
                rad_min = self.rad[i]
        self.rad_max = rad_max
        self.rad_min = rad_min

    def set_rad_colormap(self):
        for i in range(self.number-1):
            scalar = ti.min((self.rad[i] - self.rad_min) / (self.rad_max * 1.02 - self.rad_min * 0.98), 1)
            color = scalar_to_rgb(scalar)
            self.set_color(i, color)
