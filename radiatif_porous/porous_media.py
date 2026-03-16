"""
porous_media.py — Génération d'un milieu poreux 2D.

Un milieu poreux est une matrice diélectrique solide percée de trous (pores) circulaires.
  - Solide  : sigma_c fort, sigma_a variable
  - Pore    : sigma_c faible (air), sigma_a nul

Paramètres principaux :
  porosity     : fraction volumique de vide  (0 = tout solide, 1 = tout vide)
  r_mean       : rayon moyen des pores
  r_std        : écart-type du rayon (dispersion)
  seed         : graine aléatoire pour reproductibilité
  margin       : marge aux bords du domaine (évite les explosions numériques)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class PorousMedia:
    """
    Génère un milieu poreux 2D par placement aléatoire de pores circulaires.

    Attributs publics après génération :
        pores       : list de dict {'cx', 'cy', 'r'}
        mask        : np.ndarray bool (N+2, M+2), True = pore (air)
        porosity_actual : porosité réellement atteinte
    """

    def __init__(self,
                 x_min=0.0, x_max=1.0,
                 y_min=0.0, y_max=1.0,
                 porosity=0.4,
                 r_mean=0.05,
                 r_std=0.01,
                 margin=0.05,
                 max_attempts=10000,
                 allow_overlap=False,
                 seed=None):
        """
        Paramètres
        ----------
        porosity      : porosité cible  [0, 1]
        r_mean        : rayon moyen des pores
        r_std         : dispersion des rayons (0 = tous identiques)
        margin        : zone exclue près des bords (évite contact avec fantômes)
        max_attempts  : nombre max de tentatives de placement
        allow_overlap : autoriser le chevauchement des pores
        seed          : graine pour numpy.random (reproductibilité)
        """
        self.x_min = x_min;  self.x_max = x_max
        self.y_min = y_min;  self.y_max = y_max
        self.porosity    = porosity
        self.r_mean      = r_mean
        self.r_std       = r_std
        self.margin      = margin
        self.max_attempts = max_attempts
        self.allow_overlap = allow_overlap
        self.seed        = seed

        self.pores   = []
        self.mask    = None
        self.porosity_actual = 0.0

        self._rng = np.random.default_rng(seed)
        self._generate()

    # ── Génération ────────────────────────────────────────────────────────────

    def _generate(self):
        """Place des pores aléatoirement jusqu'à atteindre la porosité cible."""
        Lx = self.x_max - self.x_min
        Ly = self.y_max - self.y_min
        domain_area = Lx * Ly
        target_area = self.porosity * domain_area

        placed_area = 0.0
        attempts    = 0
        self.pores  = []

        while placed_area < target_area and attempts < self.max_attempts:
            # Rayon aléatoire (clippé pour rester positif)
            r = max(0.005, self._rng.normal(self.r_mean, self.r_std))

            # Position aléatoire avec marge
            m = self.margin + r
            if 2*m >= Lx or 2*m >= Ly:
                raise ValueError(f"Marge + rayon ({m:.3f}) trop grand pour le domaine.")

            cx = self._rng.uniform(self.x_min + m, self.x_max - m)
            cy = self._rng.uniform(self.y_min + m, self.y_max - m)

            # Vérification chevauchement
            if not self.allow_overlap:
                overlap = any(
                    np.sqrt((cx - p['cx'])**2 + (cy - p['cy'])**2) < (r + p['r'])
                    for p in self.pores
                )
                if overlap:
                    attempts += 1
                    continue

            self.pores.append({'cx': cx, 'cy': cy, 'r': r})
            placed_area += np.pi * r**2
            attempts = 0   # reset après succès

        self.porosity_actual = placed_area / domain_area
        print(f"Milieu poreux généré : {len(self.pores)} pores  |  "
              f"porosité cible={self.porosity:.2f}  "
              f"réelle={self.porosity_actual:.3f}")

    # ── Masque discret ────────────────────────────────────────────────────────

    def build_mask(self, mesh) -> np.ndarray:
        """
        Construit le masque booléen sur la grille du mesh.
        mask[i, j] = True  → pore (air)
        mask[i, j] = False → solide (diélectrique)

        Retourne un array de shape (N+2, M+2).
        """
        X, Y = np.meshgrid(mesh.x, mesh.y, indexing='ij')   # (N+2, M+2)
        mask = np.zeros(X.shape, dtype=bool)

        for p in self.pores:
            mask |= (X - p['cx'])**2 + (Y - p['cy'])**2 <= p['r']**2

        self.mask = mask
        return mask

    # ── Fonctions matériau ────────────────────────────────────────────────────

    def make_sigma_c(self, sigma_c_solid=10.0, sigma_c_air=0.5):
        """
        Retourne une fonction sigma_c_func(rho, T) → array
        compatible avec solver.sigma_c_func.
        Fonctionne quelle que soit la shape de l'input (grille complète ou intérieure).
        """
        if self.mask is None:
            raise RuntimeError("Appelez build_mask(mesh) avant make_sigma_c.")
        mask_full    = self.mask                        # (N+2, M+2)
        mask_inner   = self.mask[1:-1, 1:-1]           # (N, M)

        def sigma_c_func(rho, T):
            m = mask_inner if rho.shape == mask_inner.shape else mask_full
            return np.where(m, sigma_c_air, sigma_c_solid)
        return sigma_c_func

    def make_sigma_a(self, sigma_a_solid=0.0, sigma_a_air=0.0):
        """
        Retourne une fonction sigma_a_func(rho, T) → array.
        Fonctionne quelle que soit la shape de l'input.
        """
        if self.mask is None:
            raise RuntimeError("Appelez build_mask(mesh) avant make_sigma_a.")
        mask_full  = self.mask
        mask_inner = self.mask[1:-1, 1:-1]

        def sigma_a_func(rho, T):
            m = mask_inner if rho.shape == mask_inner.shape else mask_full
            return np.where(m, sigma_a_air, sigma_a_solid)
        return sigma_a_func

    def make_rho(self, rho_solid=10.0, rho_air=1.0):
        """Retourne une fonction rho_func(X, Y) → array."""
        def rho_func(X, Y):
            mask = np.zeros(X.shape, dtype=bool)
            for p in self.pores:
                mask |= (X - p['cx'])**2 + (Y - p['cy'])**2 <= p['r']**2
            return np.where(mask, rho_air, rho_solid)
        return rho_func

    # ── Affichage ─────────────────────────────────────────────────────────────

    def plot(self, mesh=None, ax=None, show=True):
        """
        Visualise le milieu poreux.
        Si mesh est fourni, affiche le masque discret en imshow.
        Sinon, affiche les cercles analytiques.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            fig = ax.get_figure()

        if mesh is not None and self.mask is not None:
            ax.imshow(self.mask[1:-1, 1:-1].T, origin='lower',
                      extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                      cmap='Blues', vmin=0, vmax=1, aspect='equal')
        else:
            # Fond solide
            ax.set_facecolor('#4A90D9')
            domain = plt.Rectangle((self.x_min, self.y_min),
                                    self.x_max - self.x_min,
                                    self.y_max - self.y_min,
                                    color='#4A90D9')
            ax.add_patch(domain)
            for p in self.pores:
                circle = plt.Circle((p['cx'], p['cy']), p['r'],
                                     color='white', zorder=2)
                ax.add_patch(circle)

        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect('equal')
        ax.set_title(f"Milieu poreux — {len(self.pores)} pores  "
                     f"φ={self.porosity_actual:.2f}", fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y")

        solid_patch = mpatches.Patch(color='#4A90D9', label='Solide (diélectrique)')
        air_patch   = mpatches.Patch(color='white',   label='Pore (air)',
                                      edgecolor='gray', linewidth=0.5)
        ax.legend(handles=[solid_patch, air_patch], loc='upper right', fontsize=8)

        if show:
            plt.tight_layout()
            plt.show()
        return fig, ax

    def __repr__(self):
        return (f"PorousMedia(n_pores={len(self.pores)}, "
                f"porosity_target={self.porosity:.2f}, "
                f"porosity_actual={self.porosity_actual:.3f}, "
                f"r_mean={self.r_mean}, seed={self.seed})")


# ── Test rapide ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from mesh import Mesh

    cfg = {"x_min": 0.0, "x_max": 1.0,
           "y_min": 0.0, "y_max": 1.0, "N": 60}
    mesh = Mesh(cfg)

    pm = PorousMedia(
        porosity=0.35,
        r_mean=0.06,
        r_std=0.015,
        margin=0.03,
        seed=42,
    )

    mask = pm.build_mask(mesh)
    print(f"Porosité discrète : {mask[1:-1,1:-1].mean():.3f}")
    print(pm)

    pm.plot(mesh=mesh)