import taichi as ti
import matplotlib.pyplot as plt
from render import Renderer
from format import flt_default

ti.init(arch=ti.cpu, default_fp=flt_default, debug=True)
renderer = Renderer()
# window = ti.ui.Window(name="spheres", res=(renderer.camera.resolution[1], renderer.camera.resolution[0]))
# canvas = window.get_canvas()
renderer.render()
fig = plt.figure(figsize=(renderer.camera.resolution[1]/400, renderer.camera.resolution[0]/400))
ax = fig.gca()
ax.invert_yaxis()
ax.set_axis_off()
plt.imshow(renderer.canvas.to_numpy())

fig.savefig('particleRayTracing.png', dpi=300, bbox_inches='tight', pad_inches=0)
# gui = ti.GUI("Sphere", res=(renderer.camera.resolution[1], renderer.camera.resolution[0]))
# gui.set_image(renderer.canvas_to_gui)

# while gui.running:
#     gui.set_image(renderer.canvas_to_gui)
#     gui.show()

# canvas.set_image(renderer.canvas_to_gui)
# window.save_image("particlesRayTracing.png")
# while window.running:
#     canvas.set_image(renderer.canvas_to_gui)
#     window.show()
