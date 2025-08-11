import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import numpy as np
import sys

arg1 = sys.argv[1]
freq = int(sys.argv[2])

print("Analyse du fichier :", arg1)

# === Lecture du fichier TXT ===
# Le séparateur \s+ permet de découper sur un ou plusieurs espaces
df = pd.read_csv(
    arg1,
    encoding="utf-8",
    sep=r"\s+",
    na_values=["null"],       # Convertir "null" en NaN
    engine="python"
)

df.columns = df.columns.str.replace(r"\\'b0", "°", regex=True)

# Fusionner les colonnes de date et d'heure
df['time'] = pd.to_datetime(df.iloc[:, 0] + " " + df.iloc[:, 1], errors='coerce')
df.iloc[:, 1:-1] = df.iloc[:, 2:]
df = df.drop(df.columns[-1], axis=1)
df = df.drop(df.columns[-1], axis=1)

print(df)

# === Colonnes numériques (en ignorant NaN) ===
num_cols = [
    "AccX(g)", "AccY(g)", "AccZ(g)",
    "AsX(°/s)", "AsY(°/s)", "AsZ(°/s)"
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Création de l'axe temps simulé à partir de la fréquence
dt = 1 / freq
x = [i * dt for i in range(len(df))]  # Axe des temps synthétique

# === Tracé accélérations ===
fig, ax = plt.subplots(figsize=(10,5))
lines = []
lines.append(ax.plot(x, df["AccX(g)"], 'o-', label="AccX", picker=5)[0])
lines.append(ax.plot(x, df["AccY(g)"], 'o-', label="AccY", picker=5)[0])
lines.append(ax.plot(x, df["AccZ(g)"], 'o-', label="AccZ", picker=5)[0])

plt.xlabel("Temps")
plt.ylabel("Accélération (g)")
plt.xticks(np.arange(0, max(x), 0.1))  # graduations tous les 0.1 s
plt.yticks(np.arange(int(min([min(df["AccX(g)"]), min(df["AccY(g)"]), min(df["AccZ(g)"])]))-1,
                     int(max([max(df["AccX(g)"]), max(df["AccY(g)"]), max(df["AccZ(g)"])]))+1, 0.5)) # graduations tous les 0.5 g
plt.title("Accélération mesurée par l'accéléromètre")
plt.grid(True)

# Position des cases à cocher
rax = plt.axes([0.85, 0.4, 0.1, 0.15])  # [left, bottom, width, height]
check = CheckButtons(rax, [line.get_label() for line in lines], [True]*len(lines))

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
    text = f"{line.get_label()}\nx={x_coord:.2f}, y={y_coord:.2f}"
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

check.on_clicked(func)
fig.canvas.mpl_connect("pick_event", on_pick)

plt.show()

# === Tracé vitesses angulaires ===
plt.figure(figsize=(10,5))
plt.plot(x, df["AsX(°/s)"], label="AsX")
plt.plot(x, df["AsY(°/s)"], label="AsY")
plt.plot(x, df["AsZ(°/s)"], label="AsZ")
plt.xlabel("Temps")
plt.ylabel("Vitesse angulaire (°/s)")
plt.title("Vitesse angulaire mesurée par le gyroscope")
plt.legend()
plt.grid(True)
plt.show()