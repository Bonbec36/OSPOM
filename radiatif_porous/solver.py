"""
solver.py — Transport radiatif 2D vectorisé avec NumPy.

Schéma à deux phases sur grille cartésienne uniforme :
  Phase 1 : couplage émission-absorption  (E ↔ T)
  Phase 2 : transport-diffusion           (E, Fx, Fy)

Les champs sont stockés en grilles 2D de shape (N+2, M+2),
ce qui permet d'utiliser du slicing NumPy au lieu de boucles.
Indice [i, j] → maille (i, j), avec i=0/N+1 et j=0/M+1 pour les fantômes.
"""

import numpy as np
import csv
import sys
from mesh import Mesh, cell_id


class Solver:

    def __init__(self, mesh: Mesh, cfg: dict):
        self.mesh = mesh
        N, M = mesh.N, mesh.M

        # ── Paramètres physiques ──────────────────────────────────────────────
        self.c   = float(cfg["c"])
        self.a   = float(cfg["a"])
        self.C_v = float(cfg["C_v"])

        # ── Paramètres numériques ─────────────────────────────────────────────
        self.CFL       = float(cfg["CFL"])
        self.precision = float(cfg["precision"])
        self.t_0       = float(cfg["t_0"])
        self.t_f       = float(cfg["t_f"])
        self.dt        = self.CFL * mesh.dx / self.c

        tmp = self.t_f / self.dt
        self.step_count = (int(tmp) if tmp == int(tmp) else int(tmp) + 1) + 1
        self.time_steps = np.zeros(self.step_count)

        self.save_anim = (cfg.get("export_mode") == "dataframe")

        # ── Champs 2D, shape (N+2, M+2) ──────────────────────────────────────
        # Avantage : E[i, j] est direct, les slices [1:N+1, 1:M+1] donnent
        # les mailles intérieures, [i+1,:] le voisin du haut, etc.
        self.E  = np.zeros((N+2, M+2))   # énergie radiative
        self.Fx = np.zeros((N+2, M+2))   # flux radiatif x
        self.Fy = np.zeros((N+2, M+2))   # flux radiatif y
        self.T  = np.zeros((N+2, M+2))   # température matière

        # Densité pré-calculée sur toute la grille (y compris fantômes)
        self.rho = np.zeros((N+2, M+2))

        # ── Vérifications ─────────────────────────────────────────────────────
        for name, val, cond in [
            ("c",   self.c,   self.c   > 0),
            ("a",   self.a,   self.a   > 0),
            ("C_v", self.C_v, self.C_v > 0),
            ("CFL", self.CFL, self.CFL > 0),
            ("precision", self.precision, self.precision > 0),
            ("t_f", self.t_f, self.t_f > 0),
            ("t_0", self.t_0, self.t_0 >= 0),
        ]:
            if not cond:
                raise ValueError(f"ERREUR: Vérifiez que {name} valide")

    # ── Fonctions physiques (à surcharger selon le cas) ───────────────────────

    def rho_func(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Densité du milieu. X, Y : grilles 2D de coordonnées."""
        return np.ones_like(X)

    def sigma_a_func(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Section efficace d'absorption."""
        return np.zeros_like(rho)

    def sigma_c_func(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Section efficace de scattering."""
        return np.ones_like(rho)

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(self,
                   E_0=None, Fx_0=None, Fy_0=None, T_0=None):
        """
        Initialise les champs.
        Chaque argument est soit un scalaire, soit un array (N+2, M+2),
        soit une fonction f(X, Y) → array.
        """
        mesh = self.mesh
        # Grilles de coordonnées 2D
        X, Y = np.meshgrid(mesh.x, mesh.y, indexing='ij')

        self.rho = self.rho_func(X, Y)

        self.E  = self._init_field(E_0,  X, Y, default=0.0)
        self.Fx = self._init_field(Fx_0, X, Y, default=0.0)
        self.Fy = self._init_field(Fy_0, X, Y, default=0.0)
        self.T  = self._init_field(T_0,  X, Y, default=1.0)

    @staticmethod
    def _init_field(val, X, Y, default):
        if val is None:
            return np.full_like(X, default)
        if callable(val):
            return val(X, Y)
        return np.full_like(X, float(val))

    # ── Phase 1 : couplage émission-absorption (itération point-fixe) ─────────
    # Seule phase avec une boucle : la convergence est locale par maille.

    def phase_1(self):
        c, a, C_v, dt = self.c, self.a, self.C_v, self.dt
        N, M = self.mesh.N, self.mesh.M

        # On ne traite que les mailles intérieures
        for i in range(1, N+1):
            for j in range(1, M+1):
                E_n     = self.E[i, j]
                T_n     = self.T[i, j]
                Theta_n = a * T_n**4
                E_k     = E_n
                Theta   = Theta_n

                for _ in range(1000):   # max itérations sécurité
                    T_k  = (Theta / a) ** 0.25
                    denom = T_n**3 + T_n*T_k**2 + T_k*T_n**2 + T_k**3
                    mu_q = 1.0 / denom if denom != 0 else 0.0

                    rho_k    = self.rho[i, j]
                    sig_a    = float(self.sigma_a_func(
                                    np.array([[rho_k]]),
                                    np.array([[T_k]]))[0, 0])

                    tmp1  = 1.0/dt + c*sig_a
                    alpha = (1.0/dt) / tmp1
                    beta  = c*sig_a   / tmp1

                    tmp2  = rho_k*C_v*mu_q/dt + c*sig_a
                    gamma = rho_k*C_v*mu_q/dt / tmp2
                    delta = c*sig_a            / tmp2

                    denom_pd = 1 - beta*delta
                    E_next     = (alpha*E_n + gamma*beta*Theta_n) / denom_pd
                    Theta_next = (gamma*Theta_n + alpha*delta*E_n) / denom_pd

                    if (abs(E_next - E_k) <= self.precision and
                            abs(Theta_next - Theta) <= self.precision):
                        break
                    E_k, Theta = E_next, Theta_next

                self.E[i, j] = E_next
                self.T[i, j] = (Theta_next / a) ** 0.25

    # ── Phase 2 : transport-diffusion (entièrement vectorisée) ────────────────

    def phase_2(self):
        """
        Mise à jour de E et F par le schéma de transport.
        Tout est fait avec des opérations NumPy sur les slices de la grille 2D :
          - [1:N+1, 1:M+1]  → mailles intérieures (noté I)
          - [2:N+2, 1:M+1]  → voisins droite
          - [0:N,   1:M+1]  → voisins gauche
          - [1:N+1, 2:M+2]  → voisins haut
          - [1:N+1, 0:M  ]  → voisins bas
        Aucune boucle Python.
        """
        c, dt = self.c, self.dt
        dx, dy = self.mesh.dx, self.mesh.dy
        N, M = self.mesh.N, self.mesh.M

        # Alias pour lisibilité (vues, pas de copie)
        E  = self.E
        Fx = self.Fx
        Fy = self.Fy

        # ── Sections efficaces de scattering sur toute la grille ─────────────
        sig = self.sigma_c_func(self.rho, self.T)   # shape (N+2, M+2)

        # ── Moyenne harmonique sigma_{k,l} = (sig_k + sig_l) / 2 ─────────────
        # Pour chaque direction, entre la maille intérieure et son voisin
        I = np.s_[1:N+1, 1:M+1]   # slice intérieur

        sig_up    = 0.5 * (sig[I] + sig[1:N+1, 2:M+2])   # k ↔ haut
        sig_down  = 0.5 * (sig[I] + sig[1:N+1, 0:M  ])   # k ↔ bas
        sig_left  = 0.5 * (sig[I] + sig[0:N,   1:M+1])   # k ↔ gauche
        sig_right = 0.5 * (sig[I] + sig[2:N+2, 1:M+1])   # k ↔ droite

        # ── Coefficients M_{k,l} = 2 / (2 + dx * sigma_{k,l}) ───────────────
        M_up    = 2.0 / (2.0 + dx * sig_up)
        M_down  = 2.0 / (2.0 + dx * sig_down)
        M_left  = 2.0 / (2.0 + dx * sig_left)
        M_right = 2.0 / (2.0 + dx * sig_right)

        # ── Flux de E  (n_kl · flux_E = l_kl * M_kl * (0.5*(Ek+El) - 0.5*(Fl-Fk)·n)) ──
        # Direction y (haut/bas), l_kl = dx, n = (0, ±1)
        fE_up_y   = dx * M_up   * (0.5*(E[I]+E[1:N+1,2:M+2]) - 0.5*(Fy[1:N+1,2:M+2]-Fy[I]))
        fE_down_y = dx * M_down * (0.5*(E[I]+E[1:N+1,0:M  ]) - 0.5*(Fy[I]-Fy[1:N+1,0:M  ]))  # n=(0,-1)
        # Direction x (droite/gauche), l_kl = dy, n = (±1, 0)
        fE_right_x = dy * M_right * (0.5*(E[I]+E[2:N+2,1:M+1]) - 0.5*(Fx[2:N+2,1:M+1]-Fx[I]))
        fE_left_x  = dy * M_left  * (0.5*(E[I]+E[0:N,  1:M+1]) - 0.5*(Fx[I]-Fx[0:N,  1:M+1]))  # n=(-1,0)

        # Somme des flux scalaires de F  (flux_F = l*M*(0.5*(Fk+Fl)·n - 0.5*(El-Ek)))
        fF_up    = dx * M_up    * (0.5*(Fy[I]+Fy[1:N+1,2:M+2]) - 0.5*(E[1:N+1,2:M+2]-E[I]))
        fF_down  = dx * M_down  * (-(0.5*(Fy[I]+Fy[1:N+1,0:M  ]) + 0.5*(E[1:N+1,0:M  ]-E[I])))  # n·Fy négatif
        fF_right = dy * M_right * (0.5*(Fx[I]+Fx[2:N+2,1:M+1]) - 0.5*(E[2:N+2,1:M+1]-E[I]))
        fF_left  = dy * M_left  * (-(0.5*(Fx[I]+Fx[0:N,  1:M+1]) + 0.5*(E[0:N,  1:M+1]-E[I])))

        sum_fF = fF_up + fF_down + fF_right + fF_left

        # ── Sommes nécessaires pour la mise à jour de F ───────────────────────
        sum_M_sig = (M_up*sig_up + M_down*sig_down
                   + M_left*sig_left + M_right*sig_right)

        # sum_l_M_n_x : contribution x de Σ l_kl * M_kl * n_kl
        sum_lMn_x = dy * (M_right - M_left)          # n_x = +1 (droite) ou -1 (gauche)
        sum_lMn_y = dx * (M_up    - M_down)           # n_y = +1 (haut)   ou -1 (bas)

        sum_fE_x = fE_right_x - fE_left_x             # composante x du Σ flux_E
        sum_fE_y = fE_up_y    - fE_down_y             # composante y du Σ flux_E

        # ── Mise à jour ───────────────────────────────────────────────────────
        mes_omega = dx * dy
        tmp   = 1.0/dt + c * sum_M_sig

        alpha = -c * dt / mes_omega
        beta  = (1.0/dt) / tmp
        gamma_x = (c / mes_omega) / tmp * sum_lMn_x
        gamma_y = (c / mes_omega) / tmp * sum_lMn_y
        delta = -(c / mes_omega) / tmp

        E_new  = E[I]  + alpha * sum_fF
        Fx_new = beta * Fx[I] + E[I] * gamma_x + delta * sum_fE_x
        Fy_new = beta * Fy[I] + E[I] * gamma_y + delta * sum_fE_y

        self.E [I] = E_new
        self.Fx[I] = Fx_new
        self.Fy[I] = Fy_new

    # ── Conditions aux bords ──────────────────────────────────────────────────

    def apply_boundary(self, t: float,
                       E_u=None, E_d=None, E_l=None, E_r=None,
                       Fx_u=None, Fx_d=None, Fx_l=None, Fx_r=None,
                       Fy_u=None, Fy_d=None, Fy_l=None, Fy_r=None,
                       T_u=None,  T_d=None,  T_l=None,  T_r=None):
        """
        Remplit les mailles fantômes.
        Chaque argument est soit None (→ Neumann : copie du bord intérieur),
        soit un scalaire, soit un array 1D de longueur N ou M,
        soit une fonction f(t, coords) → array.
        N, M = self.mesh.N, self.mesh.M
        """
        N, M = self.mesh.N, self.mesh.M

        def _val(spec, t, coords, interior):
            if spec is None:
                return interior          # Neumann
            if callable(spec):
                return spec(t, coords)
            return np.full_like(interior, float(spec))

        xs = self.mesh.x[1:N+1]   # coordonnées des mailles intérieures en x
        ys = self.mesh.y[1:M+1]   # idem en y

        # Haut (j = M+1), fantôme au-dessus des mailles intérieures j=M
        self.E [1:N+1, M+1] = _val(E_u,  t, xs, self.E [1:N+1, M])
        self.Fx[1:N+1, M+1] = _val(Fx_u, t, xs, self.Fx[1:N+1, M])
        self.Fy[1:N+1, M+1] = _val(Fy_u, t, xs, self.Fy[1:N+1, M])
        self.T [1:N+1, M+1] = _val(T_u,  t, xs, self.T [1:N+1, M])

        # Bas (j = 0)
        self.E [1:N+1, 0] = _val(E_d,  t, xs, self.E [1:N+1, 1])
        self.Fx[1:N+1, 0] = _val(Fx_d, t, xs, self.Fx[1:N+1, 1])
        self.Fy[1:N+1, 0] = _val(Fy_d, t, xs, self.Fy[1:N+1, 1])
        self.T [1:N+1, 0] = _val(T_d,  t, xs, self.T [1:N+1, 1])

        # Gauche (i = 0)
        self.E [0, 1:M+1] = _val(E_l,  t, ys, self.E [1, 1:M+1])
        self.Fx[0, 1:M+1] = _val(Fx_l, t, ys, self.Fx[1, 1:M+1])
        self.Fy[0, 1:M+1] = _val(Fy_l, t, ys, self.Fy[1, 1:M+1])
        self.T [0, 1:M+1] = _val(T_l,  t, ys, self.T [1, 1:M+1])

        # Droite (i = N+1)
        self.E [N+1, 1:M+1] = _val(E_r,  t, ys, self.E [N, 1:M+1])
        self.Fx[N+1, 1:M+1] = _val(Fx_r, t, ys, self.Fx[N, 1:M+1])
        self.Fy[N+1, 1:M+1] = _val(Fy_r, t, ys, self.Fy[N, 1:M+1])
        self.T [N+1, 1:M+1] = _val(T_r,  t, ys, self.T [N, 1:M+1])

    # ── Boucle principale ─────────────────────────────────────────────────────

    def solve(self, boundary_kwargs: dict = None, use_phase1: bool = True):
        """
        Lance la simulation.

        boundary_kwargs : dict passé à apply_boundary() à chaque pas de temps.
        use_phase1      : désactiver si σ_a = 0 (milieu purement diffusif).
        """
        if boundary_kwargs is None:
            boundary_kwargs = {}

        t, n = 0.0, 0
        while t <= self.t_f:
            progress = (n + 1) * 100.0 / self.step_count
            if int(progress) % 5 == 0 and (progress - int(progress)) < 0.1 and int(progress) != 0:
                print(f"  -- {int(progress):3d} %")

            if self.save_anim:
                self._save_frame(n)

            if use_phase1:
                self.phase_1()

            self.apply_boundary(t, **boundary_kwargs)
            self.phase_2()

            self.time_steps[n] = t
            t += self.dt
            n += 1

    # ── Export ────────────────────────────────────────────────────────────────

    def _save_frame(self, step: int):
        fname = f"data/anim/animation.{step}.csv"
        N, M, a = self.mesh.N, self.mesh.M, self.a
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["E", "Fx", "Fy", "T", "Tr"])
            for j in range(1, M+1):
                for i in range(1, N+1):
                    Tr = (self.E[i, j] / a) ** 0.25 if self.E[i, j] > 0 else 0.0
                    w.writerow([self.E[i,j], self.Fx[i,j], self.Fy[i,j],
                                self.T[i,j], Tr])

    def field_interior(self):
        """Retourne les champs intérieurs sous forme de vues NumPy 2D."""
        s = np.s_[1:self.mesh.N+1, 1:self.mesh.M+1]
        return self.E[s], self.Fx[s], self.Fy[s], self.T[s]