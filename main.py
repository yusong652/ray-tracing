import taichi as ti
from render import Renderer
from format import flt_default

ti.init(arch=ti.cpu, default_fp=flt_default, debug=True)
renderer = Renderer()
window = ti.ui.Window(name="spheres", res=(renderer.camera.resolution[1], renderer.camera.resolution[0]))
canvas = window.get_canvas()
renderer.render()

# gui = ti.GUI("Sphere", res=(renderer.camera.resolution[1], renderer.camera.resolution[0]))
# gui.set_image(renderer.canvas_to_gui)

# while gui.running:
#     gui.set_image(renderer.canvas_to_gui)
#     gui.show()

canvas.set_image(renderer.canvas_to_gui)
window.save_image("particlesRayTracing.png")
while window.running:
    canvas.set_image(renderer.canvas_to_gui)
    window.show()
