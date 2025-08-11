import matplotlib.pyplot as plt
import numpy as np

# Données d'exemple
x = np.linspace(0, 10, 50)
y = np.sin(x)

fig, ax = plt.subplots()
points, = ax.plot(x, y, 'o-', picker=5)  # picker=5 -> tolérance de clic en pixels

annot = ax.annotate("", xy=(0, 0), xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):
    point_index = ind[0]  # premier point sélectionné
    x_coord = x[point_index]
    y_coord = y[point_index]
    annot.xy = (x_coord, y_coord)
    text = f"x={x_coord:.2f}, y={y_coord:.2f}"
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.8)

def on_pick(event):
    if event.artist == points:
        update_annot(event.ind)
        annot.set_visible(True)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect("pick_event", on_pick)

plt.show()