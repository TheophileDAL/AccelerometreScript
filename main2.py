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

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convertit angles roll, pitch, yaw (rad) en matrice de rotation 3x3."""
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    # Ordre de rotation : Rz * Ry * Rx (yaw → pitch → roll)
    return Rz @ Ry @ Rx

# Accélérations brutes en m/s²
acc_x = df["AccX(g)"].to_numpy() * 9.81
acc_y = df["AccY(g)"].to_numpy() * 9.81
acc_z = df["AccZ(g)"].to_numpy() * 9.81
acc_raw = np.vstack((acc_x, acc_y, acc_z)).T  # shape (N,3)

# Angles en radians
roll  = np.deg2rad(df["AngleX(°)"].to_numpy())
pitch = np.deg2rad(df["AngleY(°)"].to_numpy())
yaw   = np.deg2rad(df["AngleZ(°)"].to_numpy())

g_world = np.array([0, 0, 9.81])
acc_lin = np.zeros_like(acc_raw)

for i in range(len(df)):
    R = euler_to_rotation_matrix(roll[i], pitch[i], yaw[i])
    g_sensor = R @ g_world
    acc_lin[i] = acc_raw[i] - g_sensor

print(acc_lin)

vit_x = [0.0]
vit_y = [0.0]
vit_z = [0.0]

for i in range(1, len(df)):
    vit_x.append(vit_x[-1] + (acc_lin[i][0] + acc_lin[i-1][0]) * dt * 0.5)
    vit_y.append(vit_y[-1] + (acc_lin[i][1] + acc_lin[i-1][1]) * dt * 0.5)
    vit_z.append(vit_z[-1] + (acc_lin[i][2] + acc_lin[i-1][2]) * dt * 0.5)

vit_total = np.sqrt(np.square(vit_x) + np.square(vit_y) + np.square(vit_z))

# === Tracé accélérations ===
fig, ax = plt.subplots(figsize=(10,5))
lines = []
lines.append(ax.plot(x, df["AccX(g)"], 'o-', label="AccX", picker=5)[0])
lines.append(ax.plot(x, df["AccY(g)"], 'o-', label="AccY", picker=5)[0])
lines.append(ax.plot(x, df["AccZ(g)"], 'o-', label="AccZ", picker=5)[0])

plt.xlabel("Temps")
plt.ylabel("Accélération (g)")
plt.xticks(np.arange(0, max(x), 0.2))  # graduations tous les 0.1 s
plt.yticks(np.arange(int(min([min(df["AccX(g)"]), min(df["AccY(g)"]), min(df["AccZ(g)"])]))-1,
                     int(max([max(df["AccX(g)"]), max(df["AccY(g)"]), max(df["AccZ(g)"])]))+1, 1)) # graduations tous les 1 g
plt.title("Accélération mesurée par l'accéléromètre")
plt.grid(True)

# Position des cases à cocher
rax = plt.axes([0.85, 0.4, 0.1, 0.15])  # [left, bottom, width, height]
check = CheckButtons(rax, [line.get_label() for line in lines], [True]*len(lines))

annot1 = ax.annotate("", xy=(0, 0), xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot1.set_visible(False)

annot2 = ax.annotate("", xy=(0, 0), xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot2.set_visible(False)

last_click = {'x': None, 'y': None}

def update_annot(line, ind):
    """Met à jour l'annotation avec les coordonnées du point cliqué."""
    point_index = ind[0]  # Premier point sélectionné
    xdata, ydata = line.get_data()
    x_coord = xdata[point_index]
    y_coord = ydata[point_index]
    annot1.xy = (x_coord, y_coord)
    text = f"{line.get_label()}\nx={x_coord:.3f}, y={y_coord:.3f}"
    annot1.set_text(text)
    annot1.get_bbox_patch().set_alpha(0.8)

def on_pick(event):
    """Détecte quelle courbe a été cliquée et affiche l'annotation."""
    global last_click
    x, y = event.mouseevent.x, event.mouseevent.y
    if (x, y) == (last_click['x'], last_click['y']):
        return  # ignorer le doublon
    last_click = {'x': x, 'y': y}

    if isinstance(event.artist, plt.Line2D):
        line = event.artist
        save_annot_xy = annot1.xy
        save_annot_txt = annot1.get_text()
        annot2.set_visible(annot1.get_visible())
        update_annot(line, event.ind)
        annot1.set_visible(True)
        annot2.set_text(save_annot_txt + f"\ndt={np.absolute(save_annot_xy[0] - annot1.xy[0]):.3f}")
        annot2.xy = save_annot_xy
        fig.canvas.draw_idle()

def func(label):
    index = [line.get_label() for line in lines].index(label)
    lines[index].set_visible(not lines[index].get_visible())
    plt.draw()

check.on_clicked(func)
fig.canvas.mpl_connect("pick_event", on_pick)

plt.show()

# === Tracé vitesses ===
fig, ax = plt.subplots(figsize=(10,5))
lines = []
lines.append(ax.plot(x, vit_x, 'o-', label="VitX", picker=5)[0])
lines.append(ax.plot(x, vit_y, 'o-', label="VitY", picker=5)[0])
lines.append(ax.plot(x, vit_z, 'o-', label="VitZ", picker=5)[0])
#lines.append(ax.plot(x, vit_total, 'o-', label="Vitesse", picker=5)[0])

plt.xlabel("Temps")
plt.ylabel("vitesse (m/s)")
plt.xticks(np.arange(0, max(x), 0.200))  # graduations tous les 0.2 s
plt.yticks(np.arange(int(min([min(df["AccX(g)"]), min(df["AccY(g)"]), min(df["AccZ(g)"])]))-1,
                     int(max([max(df["AccX(g)"]), max(df["AccY(g)"]), max(df["AccZ(g)"])]))+1, 1)) # graduations tous les 1 m/s
plt.title("Vitesses calculées")
plt.grid(True)

# Position des cases à cocher
rax = plt.axes([0.85, 0.4, 0.1, 0.15])  # [left, bottom, width, height]
check = CheckButtons(rax, [line.get_label() for line in lines], [True]*len(lines))

annot1 = ax.annotate("", xy=(0, 0), xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot1.set_visible(False)

annot2 = ax.annotate("", xy=(0, 0), xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot2.set_visible(False)

check.on_clicked(func)
fig.canvas.mpl_connect("pick_event", on_pick)

plt.show()
