"""
main.py — Transport radiatif 2D
Deux modes :
  "precompute"  : simule, sauvegarde toutes les frames dans un fichier .npz,
                  puis lance l'animation depuis ce fichier (fluide)
  "direct"      : simule et affiche en temps réel (plus saccadé)
  "animate"     : rejoue un fichier .npz existant sans recalculer

Usage :
  python main.py                           → mode precompute par défaut
  python main.py --mode direct             → mode direct
  python main.py --mode precompute --output frames.npz
  python main.py --mode animate   --input  frames.npz
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from mesh import Mesh
from solver import Solver
from material import rho_func, sigma_c_func, sigma_a_func


# ── Configuration ─────────────────────────────────────────────────────────────

CFG = {
    "x_min": 0.0, "x_max": 1.0,
    "y_min": 0.0, "y_max": 1.0,
    "N": 80,
    "c": 1.0, "a": 1.0, "C_v": 1.0,
    "CFL": 0.4, "precision": 1e-6,
    "t_0": 0.0, "t_f": 3.0,
    "export_mode": "none",
}

SAVE_EVERY   = 4        # 1 frame sauvegardée tous les N pas de temps
DEFAULT_FILE = "frames.npz"
INTERVAL_MS  = 40       # ms entre chaque frame dans l'animation
BOUNDARY     = {"E_l": 1.0, "Fx_l": 1.0}


# ── Construction du solveur ───────────────────────────────────────────────────

def build_solver(cfg):
    mesh = Mesh(cfg)
    solver = Solver(mesh, cfg)
    solver.rho_func     = rho_func
    solver.sigma_c_func = sigma_c_func
    solver.sigma_a_func = sigma_a_func
    solver.initialize(E_0=0.0, Fx_0=0.0, Fy_0=0.0, T_0=0.0)
    return mesh, solver


# ── Simulation ────────────────────────────────────────────────────────────────

def run_simulation(solver, save_every=SAVE_EVERY):
    """
    Lance la simulation complète avec barre de progression tqdm.
    Retourne les frames E, Fx, Fy et les temps associés.
    """
    frames_E, frames_Fx, frames_Fy = [], [], []
    saved_times = []

    solver.apply_boundary(0.0, **BOUNDARY)

    total = solver.step_count
    t, n  = 0.0, 0

    with tqdm(total=total, desc="Simulation", unit="pas",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:

        while t <= solver.t_f:
            if n % save_every == 0:
                E, Fx, Fy, _ = solver.field_interior()
                frames_E.append(E.copy())
                frames_Fx.append(Fx.copy())
                frames_Fy.append(Fy.copy())
                saved_times.append(t)

            solver.phase_2()
            solver.apply_boundary(t, **BOUNDARY)
            solver.time_steps[min(n, total - 1)] = t
            t += solver.dt
            n += 1
            pbar.update(1)

    return frames_E, frames_Fx, frames_Fy, saved_times


# ── Sauvegarde / chargement NPZ ───────────────────────────────────────────────

def save_frames(path, frames_E, frames_Fx, frames_Fy, times, cfg):
    """Sauvegarde toutes les frames dans un fichier .npz compressé."""
    np.savez_compressed(
        path,
        E    = np.array(frames_E,  dtype=np.float32),
        Fx   = np.array(frames_Fx, dtype=np.float32),
        Fy   = np.array(frames_Fy, dtype=np.float32),
        t    = np.array(times),
        x_min=cfg["x_min"], x_max=cfg["x_max"],
        y_min=cfg["y_min"], y_max=cfg["y_max"],
    )
    size_mb = os.path.getsize(path) / 1e6
    print(f"  → {len(frames_E)} frames sauvegardées dans '{path}'  ({size_mb:.1f} Mo)")


def load_frames(path):
    """Charge les frames depuis un fichier .npz."""
    data = np.load(path)
    return (
        list(data["E"]), list(data["Fx"]), list(data["Fy"]),
        list(data["t"]),
        float(data["x_min"]), float(data["x_max"]),
        float(data["y_min"]), float(data["y_max"]),
    )


# ── Animation ─────────────────────────────────────────────────────────────────

def make_animation(frames_E, frames_Fx, frames_Fy, times,
                   x_min, x_max, y_min, y_max):
    """Crée et affiche l'animation matplotlib depuis des frames pré-calculées."""

    all_frames = [frames_E, frames_Fx, frames_Fy]
    titles = ["Énergie radiative  E", "Flux  Fx", "Flux  Fy"]
    cmaps  = ["inferno", "RdBu_r", "RdBu_r"]
    extent = [x_min, x_max, y_min, y_max]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Transport radiatif 2D — disque diélectrique", fontsize=13)

    # Contour du disque (calculé une fois)
    N_, M_ = frames_E[0].shape
    xs  = np.linspace(x_min, x_max, N_)
    ys  = np.linspace(y_min, y_max, M_)
    X2D, Y2D = np.meshgrid(xs, ys, indexing='ij')
    disk_mask = ((X2D - 0.5)**2 + (Y2D - 0.5)**2 <= 0.2**2).astype(float)

    imgs = []
    for ax, title, flist, cmap in zip(axes, titles, all_frames, cmaps):
        vlo = min(f.min() for f in flist)
        vhi = max(f.max() for f in flist) or 1.0
        im  = ax.imshow(
            flist[0].T,
            origin="lower", extent=extent,
            vmin=vlo, vmax=vhi,
            cmap=cmap, aspect="equal", interpolation="bilinear",
        )
        ax.contour(X2D, Y2D, disk_mask, levels=[0.5],
                   colors="white", linewidths=1.0, linestyles="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        imgs.append(im)

    n_frames   = len(frames_E)
    time_text  = fig.text(0.5,  0.01, "", ha="center", fontsize=10, color="gray")
    frame_text = fig.text(0.01, 0.01, "", ha="left",   fontsize=9,  color="gray")

    def update(idx):
        for im, flist in zip(imgs, all_frames):
            im.set_data(flist[idx].T)
        time_text.set_text(f"t = {times[idx]:.3f}")
        frame_text.set_text(f"frame {idx + 1}/{n_frames}")

    ani = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=INTERVAL_MS,
        blit=False,
        repeat=True,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()
    return ani   # conserver la référence pour éviter le GC


# ── Modes ─────────────────────────────────────────────────────────────────────

def run_direct(cfg):
    """Simule et anime sans sauvegarder sur disque."""
    print("=== Mode DIRECT ===")
    mesh, solver = build_solver(cfg)
    print(f"Maillage : {mesh.N} × {mesh.M}  |  dt={solver.dt:.4f}  |  {solver.step_count} pas")
    frames_E, frames_Fx, frames_Fy, times = run_simulation(solver)
    print(f"{len(frames_E)} frames en mémoire — lancement de l'animation...")
    make_animation(frames_E, frames_Fx, frames_Fy, times,
                   cfg["x_min"], cfg["x_max"], cfg["y_min"], cfg["y_max"])


def run_precompute(cfg, output_path):
    """Simule, sauvegarde sur disque, puis anime."""
    print("=== Mode PRÉCALCUL ===")
    mesh, solver = build_solver(cfg)
    print(f"Maillage : {mesh.N} × {mesh.M}  |  dt={solver.dt:.4f}  |  {solver.step_count} pas")
    frames_E, frames_Fx, frames_Fy, times = run_simulation(solver)
    print("Sauvegarde des frames...")
    save_frames(output_path, frames_E, frames_Fx, frames_Fy, times, cfg)
    print("Lancement de l'animation...")
    make_animation(frames_E, frames_Fx, frames_Fy, times,
                   cfg["x_min"], cfg["x_max"], cfg["y_min"], cfg["y_max"])


def run_animate(input_path):
    """Rejoue une simulation déjà sauvegardée, sans recalculer."""
    print(f"=== Mode ANIMATION (depuis '{input_path}') ===")
    if not os.path.exists(input_path):
        print(f"Erreur : fichier '{input_path}' introuvable.")
        sys.exit(1)
    print("Chargement des frames...", end=" ", flush=True)
    frames_E, frames_Fx, frames_Fy, times, x_min, x_max, y_min, y_max = load_frames(input_path)
    print(f"{len(frames_E)} frames chargées.")
    make_animation(frames_E, frames_Fx, frames_Fy, times,
                   x_min, x_max, y_min, y_max)


# ── Point d'entrée ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transport radiatif 2D")
    parser.add_argument(
        "--mode", choices=["direct", "precompute", "animate"],
        default="precompute",
        help="Mode d'exécution (défaut : precompute)",
    )
    parser.add_argument(
        "--output", default=DEFAULT_FILE,
        help=f"Fichier .npz de sortie pour le mode precompute (défaut : {DEFAULT_FILE})",
    )
    parser.add_argument(
        "--input", default=DEFAULT_FILE,
        help=f"Fichier .npz d'entrée pour le mode animate (défaut : {DEFAULT_FILE})",
    )
    args = parser.parse_args()

    if args.mode == "direct":
        run_direct(CFG)
    elif args.mode == "precompute":
        run_precompute(CFG, args.output)
    elif args.mode == "animate":
        run_animate(args.input)