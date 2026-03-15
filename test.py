import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt

from ast import literal_eval as l_eval
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FuncAnimation

""" Chargement de la dataframe (qui contient 1 seule ligne à priori) """

converters={'rho':l_eval, 'E_u':l_eval, 'F_u':l_eval, 'T_u':l_eval, 'E_d':l_eval, 'F_d':l_eval, 'T_d':l_eval, 'E_l':l_eval, 'F_l':l_eval, 'T_l':l_eval, 'E_r':l_eval, 'F_r':l_eval, 'T_r':l_eval}
df_simu = pd.read_csv("Env_test\data\df_simu.csv", converters=converters)

x_min = df_simu.loc[0, "x_min"]
x_max = df_simu.loc[0, "x_max"]
y_min = df_simu.loc[0, "y_min"]
y_max = df_simu.loc[0, "y_max"]
t_0 = df_simu.loc[0, "t_0"]
t_f = df_simu.loc[0, "t_f"]

N = df_simu.loc[0, 'N']
M = df_simu.loc[0, 'M']
step_count = df_simu.loc[0, 'step_count']

print("x_min, x_max:", (x_min, x_max))
print("y_min, y_max:", (y_min, y_max))
print("t_0, t_f    :", (t_0, t_f))
print()
print("taille du maillage :", (N, M))
print("nombre d'itérations:", step_count)

#==================================================================================

""" Un plot de la densite """

# pour calculer les valeurs extremes d'un tenseur
def min_max(mat, dim=2):
    mat_min = mat
    for i in range(dim-1, -1, -1):
        mat_min = np.nanmin(mat_min, axis=i)
        
    mat_max = mat
    for i in range(dim-1, -1, -1):
        mat_max = np.nanmax(mat_max, axis=i)

    return mat_min, mat_max

# pour faire les plots
def plot_density(ax, df, index=0, cb=True):
    rho = np.array(df.loc[0, 'rho'])
    rho_min, rho_max = min_max(rho)
    print("(min, max) rho =", (rho_min, rho_max))

    img = ax.imshow(rho, 
                    origin='lower', 
                    cmap="viridis", 
                    interpolation='none', 
                    aspect='auto', 
                    vmin=rho_min, vmax=rho_max,
                    extent=[x_min, x_max, y_min, y_max])
    if cb == True:
        fig.colorbar(img, ax=ax)

    # set_ticks(ax)
    ax.set_title("densité", size="x-large", y=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


fig, ax = plt.subplots(1,1,figsize=(5,4))
plot_density(ax, df_simu, 0)
plt.tight_layout()
plt.show()

""" Chargement des donnnees pour le plot des signaux en video """

e = np.empty(shape=(step_count, M, N), dtype=float)
f = np.empty(shape=(step_count, M, N), dtype=float)
t = np.empty(shape=(step_count, M, N), dtype=float)
tr = np.empty(shape=(step_count, M, N), dtype=float)



for i in range(step_count):
    #file_name = "Env_test/anim/animation." + str(i) + ".csv";

    #df = pd.read_csv(file_name)
    e[i] = np.array(df['E']).reshape(M, N)
    f[i] = np.sqrt(np.array(df['F_x'])**2 + np.array(df['F_y'])**2).reshape(M, N)
    t[i] = np.array(df['T']).reshape(M, N)
    tr[i] = np.array(df['Tr']).reshape(M, N)

""" Animation de l'energie et de la norme du flux avec imshow """

fig, ax = plt.subplots(1,2,figsize=(10,4))

################### Mise en place de l'energie (ou de la temperature)
e_min, e_max = min_max(e, 3)
print("E: (min, max) =", (e_min, e_max))

t_min, t_max = min_max(t, 3)
print("T: (min, max) =", (t_min, t_max))


img1 = ax[0].imshow(e[0], origin='lower', cmap="nipy_spectral", interpolation='bilinear', extent=[x_min, x_max, y_min, y_max], vmin=e_min, vmax=e_max)
# img1 = ax[0].imshow(e[0], origin='lower', cmap="nipy_spectral", interpolation='bilinear', extent=[x_min, x_max, y_min, y_max], vmin=e_min, vmax=e_max)
# img1 = ax[0].imshow(t[0], origin='lower', cmap=cm.coolwarm, interpolation='bilinear', extent=[x_min, x_max, y_min, y_max], vmin=t_min, vmax=t_max)

fig.colorbar(img1, shrink=0.5, aspect=10, ax=ax[0])
ax[0].set_title("énergie", size="x-large")
# ax[0].set_title("température", size="x-large")

################### Mise en place du flux
f_min, f_max = min_max(f, 3)
print("F: (min, max) =", (f_min, f_max))

img2 = ax[1].imshow(f[0], origin='lower', cmap="inferno", interpolation='bilinear', extent=[x_min, x_max, y_min, y_max], vmin=f_min, vmax=f_max)
# img2 = ax[1].imshow(f[0], origin='lower', cmap="jet", interpolation='bilinear', extent=[x_min, x_max, y_min, y_max], vmin=f_min, vmax=f_max)

fig.colorbar(img2, shrink=0.5, aspect=10, ax=ax[1])
ax[1].set_title(r"intensité du flux", size="x-large")

################### Fonction d'animation
def animate(i):
    img1.set_array(e[i])    
#     img1.set_array(t[i])    

    img2.set_array(f[i])

    return [img1, img2]

anim = FuncAnimation(fig, animate, frames=step_count, repeat=False, interval=100)
plt.tight_layout()
plt.show()