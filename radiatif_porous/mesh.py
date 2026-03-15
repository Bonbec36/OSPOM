"""
mesh.py — Traduction fidèle de mesh.cpp avec NumPy.

Convention de numérotation (identique au C++) :
    k = i + j*(N+2)         ← i varie vite (direction x)
    i in [0, N+1]           ← 0 et N+1 sont les mailles fantômes gauche/droite
    j in [0, M+1]           ← 0 et M+1 sont les mailles fantômes bas/haut

Voisins (ordre fixe) :
    neighb[k, 0] = haut    (j+1)
    neighb[k, 1] = bas     (j-1)
    neighb[k, 2] = gauche  (i-1)
    neighb[k, 3] = droite  (i+1)
    -1 si le voisin est absent (bord de la grille étendue)
"""

import numpy as np


# ── Fonction identifiant de maille (miroir exact du C++) ──────────────────────

def cell_id(i, j, N2, M2):
    """
    Retourne l'identifiant linéaire de la maille (i, j).
    Fonctionne avec des scalaires ou des tableaux NumPy.
    N2 = N+2, M2 = M+2  (tailles de la grille étendue)
    """
    return i + j * N2


# ── Classe Mesh ───────────────────────────────────────────────────────────────

class Mesh:
    """
    Grille cartésienne 2D uniforme avec mailles fantômes.

    Paramètres
    ----------
    cfg : dict  avec les clés :
        x_min, x_max, y_min, y_max  (float)
        N                            (int, nombre de mailles intérieures en x)

    M est déduit automatiquement pour avoir des mailles carrées :
        M = int(N * (y_max - y_min) / (x_max - x_min))
    """

    def __init__(self, cfg: dict):
        # ── Lecture de la config ──────────────────────────────────────────────
        self.x_min = float(cfg["x_min"])
        self.x_max = float(cfg["x_max"])
        self.y_min = float(cfg["y_min"])
        self.y_max = float(cfg["y_max"])
        self.N     = int(cfg["N"])

        # ── Vérifications préliminaires ───────────────────────────────────────
        if self.x_min >= self.x_max:
            raise ValueError("ERREUR: Vérifiez que x_min < x_max")
        if self.y_min >= self.y_max:
            raise ValueError("ERREUR: Vérifiez que y_min < y_max")
        if self.N <= 0:
            raise ValueError("ERREUR: Vérifiez que N > 0")

        # ── Dimensions ───────────────────────────────────────────────────────
        N, M_ = self.N, None
        M_ = int(N * (self.y_max - self.y_min) / (self.x_max - self.x_min))
        self.M = M_
        M = self.M

        self.dx = (self.x_max - self.x_min) / N
        self.dy = (self.y_max - self.y_min) / M
        self.n_cells = (N + 2) * (M + 2)

        # ── Coordonnées des centres de mailles (y compris fantômes) ──────────
        # x[i] = x_min + (i-1)*dx + dx/2   pour i in [0, N+1]
        i_idx = np.arange(N + 2)
        j_idx = np.arange(M + 2)
        self.x = self.x_min + (i_idx - 1) * self.dx + self.dx / 2.0
        self.y = self.y_min + (j_idx - 1) * self.dy + self.dy / 2.0

        # ── Tables coord et neighb ────────────────────────────────────────────
        self._build_cells()

    # ── Construction de coord et neighb ──────────────────────────────────────

    def _build_cells(self):
        N, M = self.N, self.M
        N2, M2 = N + 2, M + 2
        n = self.n_cells

        # coord[k] = [i, j]
        # Construit vectoriellement
        i_grid, j_grid = np.meshgrid(np.arange(N2), np.arange(M2), indexing='ij')
        # k = i + j*N2  → aplatir dans le même ordre
        k_all = (i_grid + j_grid * N2).ravel()   # k_all[p] = k pour le p-ième (i,j)

        coord = np.zeros((n, 2), dtype=np.int32)
        coord[k_all, 0] = i_grid.ravel()
        coord[k_all, 1] = j_grid.ravel()
        self.coord = coord   # shape (n_cells, 2)

        # neighb : rempli d'abord pour les mailles intérieures, puis écrasé pour les bords
        neighb = np.full((n, 4), -1, dtype=np.int32)

        i_all = coord[:, 0]
        j_all = coord[:, 1]

        # Masques
        is_top    = (j_all == M + 1)
        is_bot    = (j_all == 0)
        is_left   = (i_all == 0)
        is_right  = (i_all == N + 1)
        is_inner  = ~(is_top | is_bot | is_left | is_right)

        k = np.arange(n)

        # ── Mailles fantômes du haut (j == M+1) ──────────────────────────────
        neighb[is_top, 0] = -1
        neighb[is_top, 1] = k[is_top] - N2     # voisin du bas
        neighb[is_top, 2] = -1
        neighb[is_top, 3] = -1

        # ── Mailles fantômes du bas (j == 0) ─────────────────────────────────
        neighb[is_bot, 0] = k[is_bot] + N2     # voisin du haut
        neighb[is_bot, 1] = -1
        neighb[is_bot, 2] = -1
        neighb[is_bot, 3] = -1

        # ── Mailles fantômes de gauche (i == 0) ──────────────────────────────
        neighb[is_left, 0] = -1
        neighb[is_left, 1] = -1
        neighb[is_left, 2] = -1
        neighb[is_left, 3] = k[is_left] + 1    # voisin de droite

        # ── Mailles fantômes de droite (i == N+1) ────────────────────────────
        neighb[is_right, 0] = -1
        neighb[is_right, 1] = -1
        neighb[is_right, 2] = k[is_right] - 1  # voisin de gauche
        neighb[is_right, 3] = -1

        # ── Mailles intérieures ───────────────────────────────────────────────
        neighb[is_inner, 0] = k[is_inner] + N2  # haut
        neighb[is_inner, 1] = k[is_inner] - N2  # bas
        neighb[is_inner, 2] = k[is_inner] - 1   # gauche
        neighb[is_inner, 3] = k[is_inner] + 1   # droite

        self.neighb = neighb   # shape (n_cells, 4)

    # ── Affichage (miroir de Mesh::display) ───────────────────────────────────

    def display(self):
        N, M = self.N, self.M
        N2 = N + 2
        print("-----------  Maillage  -----------")
        print("Format : (i,j) : k : (haut, bas, gauche, droite)\n")
        for j in range(M + 1, -1, -1):
            for i in range(0, N + 2):
                k = cell_id(i, j, N2, M + 2)
                ci, cj = self.coord[k]
                nb = self.neighb[k]
                print(f"  ({ci},{cj}):{k:3d}:({nb[0]:3d},{nb[1]:3d},{nb[2]:3d},{nb[3]:3d})",
                      end="")
            if j != 0:
                print("\n")
            else:
                print()

    # ── Utilitaires ───────────────────────────────────────────────────────────

    def inner_cells(self) -> np.ndarray:
        """Retourne les indices k de toutes les mailles intérieures."""
        i_all = self.coord[:, 0]
        j_all = self.coord[:, 1]
        mask = (i_all >= 1) & (i_all <= self.N) & \
               (j_all >= 1) & (j_all <= self.M)
        return np.where(mask)[0]

    def ij_to_k(self, i, j) -> np.ndarray:
        """Conversion vectorielle (i, j) → k."""
        return cell_id(np.asarray(i), np.asarray(j), self.N + 2, self.M + 2)

    def __repr__(self):
        return (f"Mesh(N={self.N}, M={self.M}, "
                f"x=[{self.x_min}, {self.x_max}], "
                f"y=[{self.y_min}, {self.y_max}], "
                f"dx={self.dx:.4f}, dy={self.dy:.4f})")


# ── Test rapide ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = {"x_min": 0.0, "x_max": 1.0,
           "y_min": 0.0, "y_max": 1.0,
           "N": 4}
    mesh = Mesh(cfg)
    print(mesh)
    print(f"n_cells = {mesh.n_cells}  (attendu : {(mesh.N+2)*(mesh.M+2)})")
    print(f"Mailles intérieures : {len(mesh.inner_cells())}  (attendu : {mesh.N*mesh.M})")
    mesh.display()