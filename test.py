import numpy as np
import matplotlib.pyplot as plt
import random

# indices optiques
N_AIR = 1.0
N_MEDIUM1 = 1.0
N_MEDIUM2 = 1.33   # exemple eau

def get_refractive_index(value):
    if value == 0:
        return N_AIR
    elif value < 0.85:
        return N_MEDIUM2
    else:
        return N_MEDIUM1
    
def random_ray(resolution):

    side = random.choice(["left", "right", "top", "bottom"])

    if side == "left":
        x = 0
        y = random.uniform(0, resolution)
        dx = random.uniform(0.5, 1)
        dy = random.uniform(-1, 1)

    elif side == "right":
        x = resolution-1
        y = random.uniform(0, resolution)
        dx = random.uniform(-1, -0.5)
        dy = random.uniform(-1, 1)

    elif side == "top":
        x = random.uniform(0, resolution)
        y = 0
        dx = random.uniform(-1, 1)
        dy = random.uniform(0.5, 1)

    else:  # bottom
        x = random.uniform(0, resolution)
        y = resolution-1
        dx = random.uniform(-1, 1)
        dy = random.uniform(-1, -0.5)

    return Ray(x, y, dx, dy)


class Ray:

    def __init__(self, x, y, dx, dy):
        self.x = x
        self.y = y

        norm = np.sqrt(dx**2 + dy**2)
        self.dx = dx / norm
        self.dy = dy / norm

        self.path_x = [x]
        self.path_y = [y]

    def trace(self, scene, step=1, max_steps=5000):

        resolution = scene.shape[0]

        for _ in range(max_steps):

            x_pix = int(self.x)
            y_pix = int(self.y)

            if x_pix < 0 or x_pix >= resolution or y_pix < 0 or y_pix >= resolution:
                break

            current_val = scene[y_pix, x_pix]
            n1 = get_refractive_index(current_val)

            # avancer
            new_x = self.x + self.dx * step
            new_y = self.y + self.dy * step

            x_pix2 = int(new_x)
            y_pix2 = int(new_y)

            if x_pix2 < 0 or x_pix2 >= resolution or y_pix2 < 0 or y_pix2 >= resolution:
                break

            next_val = scene[y_pix2, x_pix2]
            n2 = get_refractive_index(next_val)

            # changement de milieu
            if n1 != n2:

                normal = np.array([0,1])
                d = np.array([self.dx, self.dy])

                cos_i = -np.dot(normal, d)

                sin_t2 = (n1/n2)**2 * (1 - cos_i**2)

                if sin_t2 <= 1:

                    cos_t = np.sqrt(1 - sin_t2)
                    refracted = (n1/n2)*d + ( (n1/n2)*cos_i - cos_t)*normal

                    norm = np.linalg.norm(refracted)
                    self.dx = refracted[0]/norm
                    self.dy = refracted[1]/norm

            self.x = new_x
            self.y = new_y

            self.path_x.append(self.x)
            self.path_y.append(self.y)


def generate_scene(scene_size_cm=2.0, square_size_cm=1.0, wavelength_um=1.0):

    scene_size_um = scene_size_cm * 1e4
    resolution = int(scene_size_um / (10 * wavelength_um))

    scene = np.ones((resolution, resolution))

    pixels_per_cm = resolution / scene_size_cm
    square_pixels = int(square_size_cm * pixels_per_cm)

    center = resolution // 2
    half = square_pixels // 2

    x_start = center - half
    x_end = center + half
    y_start = center - half
    y_end = center + half

    scene[x_start:x_end, y_start:y_end] = 0.8

    return scene, resolution


def add_hole(scene, resolution, hole_radius_cm=0.2):

    pixels_per_cm = resolution / 2.0
    hole_radius_pixels = int(hole_radius_cm * pixels_per_cm)

    center = resolution // 2

    Y, X = np.ogrid[:resolution, :resolution]
    dist = np.sqrt((X-center)**2 + (Y-center)**2)

    scene[dist <= hole_radius_pixels] = 0

    return scene


def main():

    scene, resolution = generate_scene(scene_size_cm=1.2)
    scene = add_hole(scene, resolution, hole_radius_cm=0.4)

    ray = random_ray(resolution)
    ray.trace(scene)

    plt.imshow(scene, cmap="gray")

    plt.plot(ray.path_x, ray.path_y, color="red")

    plt.title("Ray tracing dans le medium")
    plt.show()


if __name__ == "__main__":
    main()