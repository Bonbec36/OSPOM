"""
main.py — Transport radiatif 2D — milieu poreux
Modes :
  precompute  : simule → sauvegarde .npz → anime + plot T/R/A  (défaut)
  direct      : simule + anime sans sauvegarder
  animate     : rejoue un .npz existant sans recalculer

Usage :
  python main.py
  python main.py --mode direct
  python main.py --mode animate --input frames.npz
  python main.py --mode precompute --output sim.npz
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from mesh import Mesh
from solver import Solver, NEUMANN
from porous_media import PorousMedia


# ─────────────────────────────────────────────────────────────────────────────
# Configuration du maillage et de la physique
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    "x_min": 0.0, "x_max": 1.0,
    "y_min": 0.0, "y_max": 1.0,
    "N": 80,
    "c": 1.0, "a": 1.0, "C_v": 1.0,
    "CFL": 0.4, "precision": 1e-6,
    "t_0": 0.0, "t_f": 8.0,
    "export_mode": "none",
}

# ── Milieu poreux ─────────────────────────────────────────────────────────────
POROUS_CFG = {
    "porosity":      0.45,   # fraction de vide  [0, 1]
    "r_mean":        0.06,   # rayon moyen des pores
    "r_std":         0.04,  # dispersion des rayons
    "margin":        0.05,   # marge aux bords (évite contact avec fantômes)
    "allow_overlap": True,  # chevauchement des pores autorisé ?
    "seed":          14,     # None = aléatoire à chaque run
}

# ── Propriétés optiques ───────────────────────────────────────────────────────
SIGMA_C_SOLID = 1.0   # scattering du solide diélectrique
SIGMA_C_AIR   = 0.5    # scattering de l'air (dans les pores)
SIGMA_A_SOLID = 0.01    # absorption du solide
SIGMA_A_AIR   = 0.0    # absorption de l'air

# ── Fichiers de sortie ────────────────────────────────────────────────────────
SAVE_EVERY   = 4
DEFAULT_FILE = "frames.npz"
INTERVAL_MS  = 40
ANIM_OUT     = "animation.mp4"
TRA_OUT      = "tra_plot.png"


# ─────────────────────────────────────────────────────────────────────────────
# Construction du solveur
# ─────────────────────────────────────────────────────────────────────────────

def build_solver(cfg, debug=True):
    """Construit le maillage, génère le milieu poreux et initialise le solveur."""
    mesh = Mesh(cfg)

    pm = PorousMedia(
        x_min=cfg["x_min"], x_max=cfg["x_max"],
        y_min=cfg["y_min"], y_max=cfg["y_max"],
        **POROUS_CFG,
    )
    pm.build_mask(mesh)

    # ── Visualisation debug du milieu poreux ──────────────────────────────────
    if debug:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle("Debug — Milieu poreux généré", fontsize=12)

        # Masque discret sur la grille
        mask = pm.mask[1:-1, 1:-1].T   # intérieur, orienté y vers le haut
        axes[0].imshow(mask, origin="lower", cmap="Blues",
                       extent=[cfg["x_min"], cfg["x_max"],
                                cfg["y_min"], cfg["y_max"]],
                       vmin=0, vmax=1, aspect="equal")
        axes[0].set_title(f"Masque discret  (bleu=pore/air, blanc=solide)\n"
                          f"{mask.sum()} cellules pores / {mask.size} total  "
                          f"φ={mask.mean():.3f}", fontsize=9)
        axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

        # sigma_c sur la grille
        X, Y = np.meshgrid(mesh.x[1:-1], mesh.y[1:-1], indexing='ij')
        sc_func = pm.make_sigma_c(SIGMA_C_SOLID, SIGMA_C_AIR)
        sc_field = sc_func(pm.mask[1:-1, 1:-1], None).T
        im = axes[1].imshow(sc_field, origin="lower", cmap="RdYlBu_r",
                            extent=[cfg["x_min"], cfg["x_max"],
                                    cfg["y_min"], cfg["y_max"]],
                            aspect="equal")
        axes[1].set_title(f"σ_c  (rouge={SIGMA_C_SOLID} solide, "
                          f"bleu={SIGMA_C_AIR} air)\n"
                          f"{len(pm.pores)} pores générés", fontsize=9)
        axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    solver = Solver(mesh, cfg)
    solver.rho_func     = pm.make_rho()
    solver.sigma_c_func = pm.make_sigma_c(SIGMA_C_SOLID, SIGMA_C_AIR)
    solver.sigma_a_func = pm.make_sigma_a(SIGMA_A_SOLID, SIGMA_A_AIR)
    solver.initialize(E_0=0.0, Fx_0=0.0, Fy_0=0.0, T_0=0.0)

    return mesh, solver, pm


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(solver, save_every=SAVE_EVERY):
    """Lance la simulation et retourne les frames + historique T/R/A."""
    frames_E, frames_Fx, frames_Fy = [], [], []
    times = []
    tra_history = {"T": [], "R": [], "A": [], "bilan": [], "t": []}

    solver.apply_boundary_default(0.0)
    total = solver.step_count
    t, n  = 0.0, 0

    with tqdm(total=total, desc="Simulation", unit="pas",
              bar_format="{l_bar}{bar:35}{r_bar}") as pbar:
        while t <= solver.t_f:

            if n % save_every == 0:
                E, Fx, Fy, _ = solver.field_interior()
                frames_E.append(E.copy())
                frames_Fx.append(Fx.copy())
                frames_Fy.append(Fy.copy())
                times.append(t)

            tra = solver.compute_TRA()
            tra_history["T"].append(tra["T"])
            tra_history["R"].append(tra["R"])
            tra_history["A"].append(tra["A"])
            tra_history["bilan"].append(tra["bilan"])
            tra_history["t"].append(t)

            solver.phase_2()
            solver.apply_boundary_default(t)
            solver.time_steps[min(n, total - 1)] = t
            t += solver.dt
            n += 1
            pbar.update(1)

    for k in tra_history:
        tra_history[k] = np.array(tra_history[k])

    return frames_E, frames_Fx, frames_Fy, times, tra_history


# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarde / chargement NPZ
# ─────────────────────────────────────────────────────────────────────────────

def save_frames(path, frames_E, frames_Fx, frames_Fy, times, tra, cfg):
    np.savez_compressed(
        path,
        E       = np.array(frames_E,  dtype=np.float32),
        Fx      = np.array(frames_Fx, dtype=np.float32),
        Fy      = np.array(frames_Fy, dtype=np.float32),
        t       = np.array(times),
        tra_T   = tra["T"],
        tra_R   = tra["R"],
        tra_A   = tra["A"],
        tra_bil = tra["bilan"],
        tra_t   = tra["t"],
        x_min=cfg["x_min"], x_max=cfg["x_max"],
        y_min=cfg["y_min"], y_max=cfg["y_max"],
    )
    size_mb = os.path.getsize(path) / 1e6
    print(f"  → {len(frames_E)} frames sauvegardées dans '{path}'  ({size_mb:.1f} Mo)")


def load_frames(path):
    d = np.load(path)
    tra = {
        "T":     d["tra_T"],
        "R":     d["tra_R"],
        "A":     d["tra_A"],
        "bilan": d["tra_bil"],
        "t":     d["tra_t"],
    }
    return (list(d["E"]), list(d["Fx"]), list(d["Fy"]),
            list(d["t"]), tra,
            float(d["x_min"]), float(d["x_max"]),
            float(d["y_min"]), float(d["y_max"]))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Animation des champs
# ─────────────────────────────────────────────────────────────────────────────

def make_animation(frames_E, frames_Fx, frames_Fy, times,
                   x_min, x_max, y_min, y_max,
                   pm=None, save_path=None):
    all_frames = [frames_E, frames_Fx, frames_Fy]
    titles     = ["Énergie radiative  E", "Flux  Fx", "Flux  Fy"]
    cmaps      = ["inferno", "RdBu_r", "RdBu_r"]
    extent     = [x_min, x_max, y_min, y_max]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Transport radiatif 2D — milieu poreux", fontsize=13)

    imgs = []
    for ax, title, flist, cmap in zip(axes, titles, all_frames, cmaps):
        vlo = min(f.min() for f in flist)
        vhi = max(f.max() for f in flist) or 1.0
        im  = ax.imshow(
            flist[0].T, origin="lower", extent=extent,
            vmin=vlo, vmax=vhi, cmap=cmap,
            aspect="equal", interpolation="bilinear",
        )
        # Contours des pores
        if pm is not None:
            for p in pm.pores:
                ax.add_patch(plt.Circle(
                    (p["cx"], p["cy"]), p["r"],
                    fill=False, edgecolor="white",
                    linewidth=0.6, linestyle="--",
                ))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        imgs.append(im)

    n_frames   = len(frames_E)
    time_text  = fig.text(0.5,  0.01, "", ha="center", fontsize=10, color="gray")
    frame_text = fig.text(0.01, 0.01, "", ha="left",   fontsize=9,  color="gray")

    def update(idx):
        for im, flist in zip(imgs, all_frames):
            im.set_data(flist[idx].T)
        time_text.set_text(f"t = {times[idx]:.3f}")
        frame_text.set_text(f"frame {idx+1}/{n_frames}")

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=INTERVAL_MS, blit=False, repeat=True,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        _save_animation(ani, save_path, n_frames)

    plt.show()
    return ani


def _save_animation(ani, path, n_frames):
    ext = os.path.splitext(path)[1].lower()
    print(f"Sauvegarde animation → '{path}' ...", end=" ", flush=True)
    try:
        writer = animation.PillowWriter(fps=25) if ext == ".gif" \
            else animation.FFMpegWriter(fps=25, bitrate=1800)
        ani.save(path, writer=writer,
                 progress_callback=lambda i, n: print(
                     f"\r  Encodage {i+1}/{n}   ", end="", flush=True))
        print(f"\r  → '{path}' ({os.path.getsize(path)/1e6:.1f} Mo)")
    except Exception as e:
        print(f"\n  Avertissement : impossible de sauvegarder ({e})")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — T / R / A
# ─────────────────────────────────────────────────────────────────────────────

def plot_TRA(tra, pm, save_path=None):
    t_arr = tra["t"]
    T_arr = tra["T"]
    R_arr = tra["R"]
    A_arr = tra["A"]
    B_arr = tra["bilan"]

    fig = plt.figure(figsize=(13, 5))
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1], wspace=0.35)

    # ── Courbes temporelles ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_arr, T_arr, color="#2196F3", lw=2, label="Transmission  T")
    ax1.plot(t_arr, R_arr, color="#F44336", lw=2, label="Réflexion     R")
    ax1.plot(t_arr, A_arr, color="#FF9800", lw=2, label="Absorption    A")
    ax1.plot(t_arr, B_arr, color="black",   lw=1.2, ls="--", alpha=0.7, label="Bilan")

    n_tail = max(1, len(t_arr) // 10)
    T_ss = T_arr[-n_tail:].mean()
    R_ss = R_arr[-n_tail:].mean()
    A_ss = A_arr[-n_tail:].mean()
    B_ss = B_arr[-n_tail:].mean()

    for val, col, name in [
        (T_ss, "#2196F3", f"T∞={T_ss:.3f}"),
        (R_ss, "#F44336", f"R∞={R_ss:.3f}"),
        (A_ss, "#FF9800", f"A∞={A_ss:.3f}"),
    ]:
        ax1.axhline(val, color=col, lw=0.8, ls=":", alpha=0.8)
        ax1.text(t_arr[-1] * 1.01, val, name, color=col, va="center", fontsize=8.5)

    bilan_color = "#4CAF50" if abs(B_ss - 1.0) < 0.05 else "#E53935"
    ax1.text(t_arr[-1] * 1.01, B_ss, f"Σ={B_ss:.3f}",
             color=bilan_color, va="center", fontsize=8.5, fontweight="bold")

    ax1.set_xlim(0, t_arr[-1])
    ax1.set_ylim(-0.05, 1.15)
    ax1.set_xlabel("Temps  t", fontsize=11)
    ax1.set_ylabel("Fraction du flux incident", fontsize=11)
    ax1.set_title("Évolution temporelle de T, R, A", fontsize=11)
    ax1.legend(loc="center right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Schéma annoté ─────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], aspect="equal")
    ax2.set_xlim(-0.3, 1.3)
    ax2.set_ylim(-0.2, 1.2)
    ax2.axis("off")
    ax2.set_title("Bilan énergétique  (état stationnaire)", fontsize=11)

    ax2.add_patch(plt.Rectangle((0, 0), 1, 1, lw=1.5,
                                 edgecolor="gray", facecolor="#4A90D9", alpha=0.6))
    for p in pm.pores:
        ax2.add_patch(plt.Circle((p["cx"], p["cy"]), p["r"],
                                  facecolor="white", edgecolor="#1565C0",
                                  lw=0.5, alpha=0.8))
    ax2.text(0.5, 0.5, f"φ={pm.porosity_actual:.2f}\n{len(pm.pores)} pores",
             ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    def arrow(ax, x0, y0, dx, dy, color, label):
        ax.annotate("", xy=(x0+dx, y0+dy), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5, mutation_scale=14))
        ax.text(x0+dx/2, y0+dy/2+0.05, label,
                ha="center", fontsize=9, color=color, fontweight="bold")

    arrow(ax2, -0.28, 0.5,  0.2,  0,  "#555",    "I = 1.0")
    arrow(ax2,  1.02, 0.5,  0.2,  0,  "#2196F3", f"T={T_ss:.3f}")
    arrow(ax2,  0.02, 0.54, -0.2, 0,  "#F44336", f"R={R_ss:.3f}")
    ax2.annotate("", xy=(0.5, -0.15), xytext=(0.5, 0.02),
                 arrowprops=dict(arrowstyle="-|>", color="#FF9800", lw=2.5, mutation_scale=14))
    ax2.text(0.5, -0.19, f"A={A_ss:.3f}",
             ha="center", fontsize=9, color="#FF9800", fontweight="bold")

    bilan_color = "#4CAF50" if abs(B_ss - 1.0) < 0.05 else "#E53935"
    ax2.text(0.5, 1.12, f"Bilan  T+R+A = {B_ss:.3f}",
             ha="center", fontsize=9, color=bilan_color,
             style="italic", fontweight="bold")

    plt.suptitle("Analyse optique — milieu poreux", fontsize=13, y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Figure T/R/A sauvegardée dans '{save_path}'")

    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────────

def run_direct(cfg):
    print("=== Mode DIRECT ===")
    mesh, solver, pm = build_solver(cfg)
    print(f"Maillage : {mesh.N} × {mesh.M}  |  dt={solver.dt:.4f}  |  {solver.step_count} pas")
    frames_E, frames_Fx, frames_Fy, times, tra = run_simulation(solver)
    make_animation(frames_E, frames_Fx, frames_Fy, times,
                   cfg["x_min"], cfg["x_max"], cfg["y_min"], cfg["y_max"],
                   pm=pm, save_path=ANIM_OUT)
    plot_TRA(tra, pm, save_path=TRA_OUT)


def run_precompute(cfg, output_path):
    print("=== Mode PRÉCALCUL ===")
    mesh, solver, pm = build_solver(cfg)
    print(f"Maillage : {mesh.N} × {mesh.M}  |  dt={solver.dt:.4f}  |  {solver.step_count} pas")
    frames_E, frames_Fx, frames_Fy, times, tra = run_simulation(solver)
    print("Sauvegarde des données...")
    save_frames(output_path, frames_E, frames_Fx, frames_Fy, times, tra, cfg)
    make_animation(frames_E, frames_Fx, frames_Fy, times,
                   cfg["x_min"], cfg["x_max"], cfg["y_min"], cfg["y_max"],
                   pm=pm, save_path=ANIM_OUT)
    plot_TRA(tra, pm, save_path=TRA_OUT)


def run_animate(input_path):
    print(f"=== Mode ANIMATION (depuis '{input_path}') ===")
    if not os.path.exists(input_path):
        print(f"Erreur : fichier '{input_path}' introuvable.")
        sys.exit(1)
    print("Chargement...", end=" ", flush=True)
    frames_E, frames_Fx, frames_Fy, times, tra, x_min, x_max, y_min, y_max = load_frames(input_path)
    print(f"{len(frames_E)} frames chargées.")
    pm = PorousMedia(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, **POROUS_CFG)
    make_animation(frames_E, frames_Fx, frames_Fy, times,
                   x_min, x_max, y_min, y_max,
                   pm=pm, save_path=ANIM_OUT)
    plot_TRA(tra, pm, save_path=TRA_OUT)


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transport radiatif 2D — milieu poreux")
    parser.add_argument("--mode", choices=["direct", "precompute", "animate"],
                        default="precompute")
    parser.add_argument("--output", default=DEFAULT_FILE)
    parser.add_argument("--input",  default=DEFAULT_FILE)
    args = parser.parse_args()

    if args.mode == "direct":
        run_direct(CFG)
    elif args.mode == "precompute":
        run_precompute(CFG, args.output)
    elif args.mode == "animate":
        run_animate(args.input)