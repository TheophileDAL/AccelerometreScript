

annot = ax.annotate("", xy=(0, 0), xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(line, ind):
    """Met à jour l'annotation avec les coordonnées du point cliqué."""
    point_index = ind[0]  # Premier point sélectionné
    xdata, ydata = line.get_data()
    x_coord = xdata[point_index]
    y_coord = ydata[point_index]
    annot.xy = (x_coord, y_coord)
    text = f"{line.get_label()}\nx={x_coord:.3f}, y={y_coord:.3f}"
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.8)

def on_pick(event):
    """Détecte quelle courbe a été cliquée et affiche l'annotation."""
    if isinstance(event.artist, plt.Line2D):
        line = event.artist
        update_annot(line, event.ind)
        annot.set_visible(True)
        fig.canvas.draw_idle()

def func(label):
    index = [line.get_label() for line in lines].index(label)
    lines[index].set_visible(not lines[index].get_visible())
    plt.draw()