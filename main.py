import taichi as ti
import matplotlib.pyplot as plt
from render import Renderer
from format import flt_default

def main():
    ti.init(arch=ti.cpu, device_memory_fraction=1.0, default_fp=flt_default, debug=True)
    renderer = Renderer()
    resolution = (renderer.camera.resolution[1], renderer.camera.resolution[0])
    window = ti.ui.Window("Ball in space", resolution)
    canvas = window.get_canvas()
    supersample = 4
    while window.running:
        renderer.render(supersample)
        canvas.set_image(renderer.canvas_to_gui)
        window.show()
    fig = plt.figure(figsize=(renderer.camera.resolution[1]/400, renderer.camera.resolution[0]/400))
    ax = fig.gca()
    ax.set_axis_off()
    plt.imshow(renderer.canvas.to_numpy())
    plt.gca().invert_yaxis()
    fig.savefig('particleRayTracing.png', dpi=500, bbox_inches='tight', pad_inches=0)
    
if __name__ == '__main__':
    main()
