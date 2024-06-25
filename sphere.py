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

    @ti.func
    def set_color(self, i: ti.i32, color: vec):
        self.color[i, 0] = color[0]
        self.color[i, 1] = color[1]
        self.color[i, 2] = color[2]

    @ti.func
    def set_specular(self, i: ti.i32, specular: flt_default):
        self.specular[i] = specular

    def load_file(self):
        try:
            file_name = 'ball_info_501.csv'
            df = pd.read_csv(file_name)
            self.number = df.shape[0]
            self.rad = ti.field(dtype=flt_default, shape=(self.number,))
            self.pos = ti.field(dtype=flt_default, shape=(self.number, 3))
            self.color = ti.field(dtype=flt_default, shape=(self.number, 3))
            self.specular = ti.field(dtype=flt_default, shape=(self.number,))
            self.reflective = ti.field(dtype=flt_default, shape=(self.number,))
            radii = df['rad'].to_numpy()
            pos_x = df['pos_x'].to_numpy()
            pos_y = df['pos_y'].to_numpy()
            pos_z = df['pos_z'].to_numpy()
            # specular = df['specular'].to_numpy()
            for i in range(radii.shape[0]):
                self.rad[i] = radii[i]
                self.pos[i, 0] = pos_x[i]
                self.pos[i, 1] = pos_y[i]
                self.pos[i, 2] = pos_z[i]
                self.specular[i] = 64
                self.reflective[i] = 0.5

        except FileNotFoundError:
            self.default_init()

    def default_init(self):
        self.number = 2
        self.rad = ti.field(dtype=flt_default, shape=(self.number,))
        self.pos = ti.field(dtype=flt_default, shape=(self.number, 3))
        self.color = ti.field(dtype=flt_default, shape=(self.number, 3))
        self.specular = ti.field(dtype=flt_default, shape=(self.number,))
        self.reflective = ti.field(dtype=flt_default, shape=(self.number,))
        self.rad[0] = 0.28
        self.pos[0, 0] = 1.5
        self.pos[0, 1] = -0.3
        self.pos[0, 2] = -0.3
        self.specular[0] = 16.0
        self.reflective[0] = 0.4
        self.rad[1] = 0.26
        self.pos[1, 0] = 1.5
        self.pos[1, 1] = -0.3
        self.pos[1, 2] = 0.3
        self.specular[1] = 64
        self.reflective[1] = 0.4

    @ti.func
    def get_pos(self, i: ti.i32):
        return vec(self.pos[i, 0], self.pos[i, 1], self.pos[i, 2])

    @ti.func
    def get_color(self, i: ti.i32):
        return vec(self.color[i, 0], self.color[i, 1], self.color[i, 2])

    def get_rad_range(self):
        rad_max = self.rad[0]
        rad_min = self.rad[0]
        for i in range(self.number):
            if self.rad[i] >= rad_max:
                rad_max = self.rad[i]
            if self.rad[i] <= rad_min:
                rad_min = self.rad[i]
        self.rad_max = rad_max
        self.rad_min = rad_min

    def set_rad_colormap(self):
        for i in range(self.number):
            scalar = (self.rad[i] - self.rad_min) / (self.rad_max * 1.02 - self.rad_min * 0.98)
            self.color[i, 0] = scalar_to_rgb(scalar)[0]
            self.color[i, 1] = scalar_to_rgb(scalar)[1]
            self.color[i, 2] = scalar_to_rgb(scalar)[2]