"""
main.py — Transport radiatif 2D
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
from material import rho_func, sigma_c_func, sigma_a_func, DISK_CX, DISK_CY, DISK_R

# ── Configuration ─────────────────────────────────────────────────────────────

CFG = {
    "x_min": 0.0, "x_max": 1.0,
    "y_min": 0.0, "y_max": 1.0,
    "N": 80,
    "c": 1.0, "a": 1.0, "C_v": 1.0,
    "CFL": 0.4, "precision": 1e-6,
    "t_0": 0.0, "t_f": 15.0,
    "export_mode": "none",
}

SAVE_EVERY    = 4
DEFAULT_FILE  = "frames.npz"
INTERVAL_MS   = 40
ANIM_OUT      = "animation.mp4"
TRA_OUT       = "tra_plot.png"


# ── Construction du solveur ───────────────────────────────────────────────────

def build_solver(cfg):
    mesh   = Mesh(cfg)
    solver = Solver(mesh, cfg)
    solver.rho_func     = rho_func
    solver.sigma_c_func = sigma_c_func
    solver.sigma_a_func = sigma_a_func
    solver.initialize(E_0=0.0, Fx_0=0.0, Fy_0=0.0, T_0=0.0)
    return mesh, solver


# ── Simulation ────────────────────────────────────────────────────────────────

def run_simulation(solver, save_every=SAVE_EVERY):
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
            tra_history["bilan"].append(tra["T"] + tra["R"] + tra["A"])
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


# ── Sauvegarde / chargement NPZ ───────────────────────────────────────────────

def save_frames(path, frames_E, frames_Fx, frames_Fy, times, tra_history, cfg):
    np.savez_compressed(
        path,
        E        = np.array(frames_E,  dtype=np.float32),
        Fx       = np.array(frames_Fx, dtype=np.float32),
        Fy       = np.array(frames_Fy, dtype=np.float32),
        t        = np.array(times),
        tra_T    = tra_history["T"],
        tra_R    = tra_history["R"],
        tra_A    = tra_history["A"],
        tra_bil  = tra_history["bilan"],
        tra_t    = tra_history["t"],
        x_min=cfg["x_min"], x_max=cfg["x_max"],
        y_min=cfg["y_min"], y_max=cfg["y_max"],
    )
    size_mb = os.path.getsize(path) / 1e6
    print(f"  → {len(frames_E)} frames sauvegardées dans '{path}'  ({size_mb:.1f} Mo)")


def load_frames(path):
    d = np.load(path)
    tra_history = {
        "T":     d["tra_T"],
        "R":     d["tra_R"],
        "A":     d["tra_A"],
        "bilan": d["tra_bil"],
        "t":     d["tra_t"],
    }
    return (list(d["E"]), list(d["Fx"]), list(d["Fy"]),
            list(d["t"]), tra_history,
            float(d["x_min"]), float(d["x_max"]),
            float(d["y_min"]), float(d["y_max"]))


# ── Figure 1 — Animation des champs ──────────────────────────────────────────

def make_animation(frames_E, frames_Fx, frames_Fy, times,
                   x_min, x_max, y_min, y_max,
                   save_path=None):
    """
    Anime E, Fx, Fy en imshow.
    Si save_path est fourni, sauvegarde la vidéo/gif ET affiche la fenêtre.
    """
    all_frames = [frames_E, frames_Fx, frames_Fy]
    titles     = ["Énergie radiative  E", "Flux  Fx", "Flux  Fy"]
    cmaps      = ["inferno", "RdBu_r", "RdBu_r"]
    extent     = [x_min, x_max, y_min, y_max]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Transport radiatif 2D — disque diélectrique", fontsize=13)

    # Contour du disque
    N_, M_ = frames_E[0].shape
    xs  = np.linspace(x_min, x_max, N_)
    ys  = np.linspace(y_min, y_max, M_)
    X2D, Y2D = np.meshgrid(xs, ys, indexing='ij')
    disk_mask = ((X2D - DISK_CX)**2 + (Y2D - DISK_CY)**2 <= DISK_R**2).astype(float)

    imgs = []
    for ax, title, flist, cmap in zip(axes, titles, all_frames, cmaps):
        vlo = min(f.min() for f in flist)
        vhi = max(f.max() for f in flist) or 1.0
        im  = ax.imshow(
            flist[0].T, origin="lower", extent=extent,
            vmin=vlo, vmax=vhi, cmap=cmap,
            aspect="equal", interpolation="bilinear",
        )
        ax.contour(X2D, Y2D, disk_mask, levels=[0.5],
                   colors="white", linewidths=1.0, linestyles="--")
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
    """Sauvegarde l'animation en mp4 (ffmpeg) ou gif (Pillow) selon l'extension."""
    ext = os.path.splitext(path)[1].lower()
    print(f"Sauvegarde de l'animation → '{path}' ...", end=" ", flush=True)
    try:
        if ext == ".gif":
            writer = animation.PillowWriter(fps=25)
        else:
            writer = animation.FFMpegWriter(fps=25, bitrate=1800)
        ani.save(path, writer=writer,
                 progress_callback=lambda i, n: print(f"\r  Encodage {i+1}/{n}   ",
                                                       end="", flush=True))
        print(f"\r  → '{path}' sauvegardé ({os.path.getsize(path)/1e6:.1f} Mo)")
    except Exception as e:
        print(f"\n  Avertissement : impossible de sauvegarder l'animation ({e})")
        print("  Installez ffmpeg (mp4) ou Pillow (gif) pour activer l'export vidéo.")


# ── Figure 2 — Transmission / Réflexion / Absorption ─────────────────────────

def plot_TRA(tra_history, mesh, save_path=None):
    """
    Deux sous-figures :
      Gauche  : T(t), R(t), A(t) et bilan complet T+R+A+F_top+F_bot
      Droite  : schéma annoté du domaine avec les flux à t_final
    """
    t_arr   = tra_history["t"]
    T_arr   = tra_history["T"]
    R_arr   = tra_history["R"]
    A_arr   = tra_history["A"]
    Ft_arr  = tra_history.get("F_top", np.zeros_like(T_arr))
    Fb_arr  = tra_history.get("F_bot", np.zeros_like(T_arr))
    B_arr   = T_arr + R_arr + A_arr + Ft_arr + Fb_arr

    fig = plt.figure(figsize=(13, 5))
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1], wspace=0.35)

    # ── Sous-figure gauche : courbes temporelles ───────────────────────────────
    ax1 = fig.add_subplot(gs[0])

    ax1.plot(t_arr, T_arr,  color="#2196F3", lw=2,   label="Transmission  T")
    ax1.plot(t_arr, R_arr,  color="#F44336", lw=2,   label="Réflexion     R")
    ax1.plot(t_arr, A_arr,  color="#FF9800", lw=2,   label="Absorption    A")
    if np.any(Ft_arr > 1e-4) or np.any(Fb_arr > 1e-4):
        ax1.plot(t_arr, Ft_arr, color="#9C27B0", lw=1.5, ls="-.",
                 label="Fuite haut  F↑")
        ax1.plot(t_arr, Fb_arr, color="#009688", lw=1.5, ls="-.",
                 label="Fuite bas   F↓")
    ax1.plot(t_arr, B_arr,  color="black",  lw=1.2,
             ls="--", alpha=0.7, label="Bilan total")

    # Valeurs stationnaires (moyenne des 10 derniers %)
    n_tail = max(1, len(t_arr) // 10)
    T_ss  = T_arr [-n_tail:].mean()
    R_ss  = R_arr [-n_tail:].mean()
    A_ss  = A_arr [-n_tail:].mean()
    Ft_ss = Ft_arr[-n_tail:].mean()
    Fb_ss = Fb_arr[-n_tail:].mean()
    B_ss  = T_ss + R_ss + A_ss + Ft_ss + Fb_ss

    for val, col, name in [
        (T_ss, "#2196F3", f"T∞={T_ss:.3f}"),
        (R_ss, "#F44336", f"R∞={R_ss:.3f}"),
        (A_ss, "#FF9800", f"A∞={A_ss:.3f}"),
    ]:
        ax1.axhline(val, color=col, lw=0.8, ls=":", alpha=0.8)
        ax1.text(t_arr[-1]*1.01, val, name, color=col,
                 va="center", fontsize=8.5)

    bilan_color = "#4CAF50" if abs(B_ss - 1.0) < 0.05 else "#E53935"
    ax1.axhline(B_ss, color=bilan_color, lw=0.8, ls=":", alpha=0.8)
    ax1.text(t_arr[-1]*1.01, B_ss, f"Σ={B_ss:.3f}",
             color=bilan_color, va="center", fontsize=8.5, fontweight="bold")

    ax1.set_xlim(0, t_arr[-1])
    ax1.set_ylim(-0.05, 1.15)
    ax1.set_xlabel("Temps  t", fontsize=11)
    ax1.set_ylabel("Fraction du flux incident", fontsize=11)
    ax1.set_title("Évolution temporelle de T, R, A", fontsize=11)
    ax1.legend(loc="center right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Sous-figure droite : schéma annoté du domaine ─────────────────────────
    ax2 = fig.add_subplot(gs[1], aspect="equal")
    ax2.set_xlim(-0.3, 1.3)
    ax2.set_ylim(-0.2, 1.2)
    ax2.axis("off")
    ax2.set_title("Bilan énergétique  (état stationnaire)", fontsize=11)

    domain = plt.Rectangle((0, 0), 1, 1, lw=1.5,
                            edgecolor="gray", facecolor="#f0f4f8")
    ax2.add_patch(domain)

    disk = plt.Circle((DISK_CX, DISK_CY), DISK_R,
                       facecolor="#90CAF9", edgecolor="#1565C0", lw=1.5)
    ax2.add_patch(disk)
    ax2.text(DISK_CX, DISK_CY, "Disque\ndiélectrique",
             ha="center", va="center", fontsize=7.5, color="#0D47A1")

    def arrow(ax, x0, y0, dx, dy, color, label):
        ax.annotate("", xy=(x0+dx, y0+dy), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=2.5, mutation_scale=14))
        ax.text(x0+dx/2, y0+dy/2+0.05, label,
                ha="center", fontsize=9, color=color, fontweight="bold")

    arrow(ax2, -0.28, 0.5,  0.2,  0,    "#555",    "I = 1.0")
    arrow(ax2,  1.0,  0.5,  0.2,  0,    "#2196F3", f"T={T_ss:.3f}")
    arrow(ax2,  0.02, 0.54, -0.2, 0,    "#F44336", f"R={R_ss:.3f}")
    # Absorption vers le bas
    ax2.annotate("", xy=(0.5, -0.15), xytext=(0.5, DISK_CY-DISK_R),
                 arrowprops=dict(arrowstyle="-|>", color="#FF9800",
                                  lw=2.5, mutation_scale=14))
    ax2.text(0.5, -0.19, f"A={A_ss:.3f}",
             ha="center", fontsize=9, color="#FF9800", fontweight="bold")

    # Fuites haut/bas si significatives
    if Ft_ss > 1e-3:
        arrow(ax2, 0.5, 1.0, 0, 0.15, "#9C27B0", f"↑{Ft_ss:.3f}")
    if Fb_ss > 1e-3:
        arrow(ax2, 0.5, 0.0, 0, -0.15, "#009688", f"↓{Fb_ss:.3f}")

    bilan_color = "#4CAF50" if abs(B_ss - 1.0) < 0.05 else "#E53935"
    ax2.text(0.5, 1.15, f"Bilan  T+R+A = {B_ss:.3f}",
             ha="center", fontsize=9, color=bilan_color,
             style="italic", fontweight="bold")

    plt.suptitle("Analyse optique — milieu poreux", fontsize=13, y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Figure T/R/A sauvegardée dans '{save_path}'")

    plt.show()
    return fig


# ── Modes ─────────────────────────────────────────────────────────────────────

def run_direct(cfg):
    print("=== Mode DIRECT ===")
    mesh, solver = build_solver(cfg)
    print(f"Maillage : {mesh.N} × {mesh.M}  |  dt={solver.dt:.4f}  |  {solver.step_count} pas")

    frames_E, frames_Fx, frames_Fy, times, tra = run_simulation(solver)

    make_animation(frames_E, frames_Fx, frames_Fy, times,
                   cfg["x_min"], cfg["x_max"], cfg["y_min"], cfg["y_max"],
                   save_path=ANIM_OUT)
    plot_TRA(tra, mesh, save_path=TRA_OUT)


def run_precompute(cfg, output_path):
    print("=== Mode PRÉCALCUL ===")
    mesh, solver = build_solver(cfg)
    print(f"Maillage : {mesh.N} × {mesh.M}  |  dt={solver.dt:.4f}  |  {solver.step_count} pas")

    frames_E, frames_Fx, frames_Fy, times, tra = run_simulation(solver)

    print("Sauvegarde des données...")
    save_frames(output_path, frames_E, frames_Fx, frames_Fy, times, tra, cfg)

    ani = make_animation(frames_E, frames_Fx, frames_Fy, times,
                         cfg["x_min"], cfg["x_max"], cfg["y_min"], cfg["y_max"],
                         save_path=ANIM_OUT)
    plot_TRA(tra, mesh, save_path=TRA_OUT)


def run_animate(input_path):
    print(f"=== Mode ANIMATION (depuis '{input_path}') ===")
    if not os.path.exists(input_path):
        print(f"Erreur : fichier '{input_path}' introuvable.")
        sys.exit(1)

    print("Chargement...", end=" ", flush=True)
    frames_E, frames_Fx, frames_Fy, times, tra, x_min, x_max, y_min, y_max = load_frames(input_path)
    print(f"{len(frames_E)} frames chargées.")

    # Mesh factice pour plot_TRA (juste besoin des dimensions)
    cfg_tmp = {**CFG, "x_min": x_min, "x_max": x_max,
               "y_min": y_min, "y_max": y_max}
    mesh, _ = build_solver(cfg_tmp)

    ani = make_animation(frames_E, frames_Fx, frames_Fy, times,
                         x_min, x_max, y_min, y_max,
                         save_path=ANIM_OUT)
    plot_TRA(tra, mesh, save_path=TRA_OUT)


# ── Point d'entrée ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transport radiatif 2D")
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