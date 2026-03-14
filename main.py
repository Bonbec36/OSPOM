import numpy as np
import matplotlib.pyplot as plt

def generate_scene(scene_size_cm=2.0, square_size_cm=1.0, wavelength_um=1.0):
    """
    Génère une scène 2 cm x 2 cm avec un carré central gris clair.
    La résolution est limitée à 10× la longueur d'onde.
    """
    # calcul de la résolution maximale
    scene_size_um = scene_size_cm * 1e4  # 1 cm = 10000 µm
    max_resolution = int(scene_size_um / (10 * wavelength_um))
    resolution = max_resolution

    # matrice de fond (blanc)
    scene = np.ones((resolution, resolution))

    # conversion cm → pixels
    pixels_per_cm = resolution / scene_size_cm
    square_pixels = int(square_size_cm * pixels_per_cm)

    # position du carré (centré)
    center = resolution // 2
    half = square_pixels // 2
    x_start = center - half
    x_end = center + half
    y_start = center - half
    y_end = center + half

    # carré gris clair
    scene[x_start:x_end, y_start:y_end] = 0.8

    return scene, resolution

def add_hole(scene, resolution, hole_radius_cm=0.2):
    """
    Ajoute un trou noir circulaire au centre de la scène.
    """
    pixels_per_cm = resolution / 2.0  # scene_size_cm = 2 cm
    hole_radius_pixels = int(hole_radius_cm * pixels_per_cm)
    center = resolution // 2

    Y, X = np.ogrid[:resolution, :resolution]
    dist_from_center = np.sqrt((X - center)**2 + (Y - center)**2)
    scene[dist_from_center <= hole_radius_pixels] = 0.0  # noir

    return scene

def main():
    scene, resolution = generate_scene(scene_size_cm=1.2, square_size_cm=1.0, wavelength_um=1.0)
    scene = add_hole(scene, resolution, hole_radius_cm=0.4)

    plt.imshow(scene, cmap="gray", extent=[0,2,0,2])
    plt.title("Scène 2 cm x 2 cm : carré gris clair avec trou noir central")
    plt.xlabel("cm")
    plt.ylabel("cm")
    plt.show()

if __name__ == "__main__":
    main()