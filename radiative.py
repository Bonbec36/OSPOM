import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =====================
# Création de la grille
# =====================
grid_points = 50
scene_size = 1.5e-2
square_size = 1e-2

x = np.linspace(0, scene_size, grid_points)
y = np.linspace(0, scene_size, grid_points)
X, Y = np.meshgrid(x, y)

# Matrice matière (pour visualisation)
matter_matrix = np.zeros((grid_points, grid_points))
square_start = (scene_size - square_size)/2
square_end = (scene_size + square_size)/2
mask_square = (X >= square_start) & (X <= square_end) & (Y >= square_start) & (Y <= square_end)
matter_matrix[mask_square] = 1.0  # carré central = 1

# Matrice intensité
I = np.zeros((grid_points, grid_points))

# Source ponctuelle (ex. à gauche au milieu)
source_x = 0
source_y = grid_points // 2

# Matrices de coefficients
absorption_matrix = np.zeros((grid_points, grid_points))
diffusion_matrix  = np.zeros((grid_points, grid_points))
absorption_matrix[:] = 0.0
diffusion_matrix[:]  = 0.1
absorption_matrix[mask_square] = 0.1
diffusion_matrix[mask_square]  = 0.01

# =====================
# Fonction de mise à jour (cellule par cellule)
# =====================
def update(frame):
    global I
    
    I_new = I.copy()
    
    # Source
    I_new[source_y, source_x] = 1.0
    
    # =====================
    # Propagation principale (de gauche à droite)
    # =====================
    for i in range(grid_points):
        for j in range(grid_points-1):  # propagation vers la droite
            I_new[i, j+1] += 0.5 * I[i,j]  # transfert partiel
            I_new[i, j]   *= 0.5           # perte dans la cellule après propagation
    
    # =====================
    # Diffusion vers 4 voisins et absorption
    # =====================
    for i in range(1, grid_points-1):
        for j in range(1, grid_points-1):
            diffusion = diffusion_matrix[i,j] * (
                I[i-1,j] + I[i+1,j] + I[i,j-1] + I[i,j+1] - 4*I[i,j]
            )
            absorption = -absorption_matrix[i,j] * I[i,j]
            I_new[i,j] += diffusion + absorption
    
    I[:] = I_new
    im.set_array(I)
    return [im]

# =====================
# Animation
# =====================
fig, axs = plt.subplots(1, 2, figsize=(12,6))

# Affichage matière
im1 = axs[0].imshow(matter_matrix, cmap='plasma', origin='lower', extent=[0, scene_size*100, 0, scene_size*100])
axs[0].set_title("Matrice matière")
axs[0].set_xlabel("x (cm)")
axs[0].set_ylabel("y (cm)")
plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04, label="Valeur matière")

# Affichage intensité
im = axs[1].imshow(I, cmap='viridis', origin='lower', extent=[0, scene_size*100, 0, scene_size*100], vmin=0, vmax=1)
axs[1].set_title("Intensité (champ I)")
axs[1].set_xlabel("x (cm)")
axs[1].set_ylabel("y (cm)")
plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04, label="Intensité")

plt.tight_layout()

# Animation
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)
plt.show()