import taichi as ti
import matplotlib.pyplot as plt
vec = ti.math.vec3


def scalar_to_rgb(scalar: ti.f32) -> vec:
    """
    convert scalar ranging from 0.0 to 1.0 to rgb
    :param scalar:
    :return:
    """
    if scalar < 0 or scalar > 1:
        raise ValueError("Scalar value must be within the range [0, 1].")
    colormap = plt.get_cmap('rainbow')
    rgb = colormap(scalar)

    return vec(rgb[0], rgb[1], rgb[2])

