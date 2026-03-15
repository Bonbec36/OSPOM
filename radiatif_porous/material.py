"""
material.py — Propriétés optiques du milieu.

Cas test : disque diélectrique centré dans un carré d'air.
  - Air      : sigma_c faible, sigma_a nul
  - Disque   : sigma_c fort (diffusion), sigma_a nul (transparent)
"""

import numpy as np


def in_disk(X, Y, cx=0.5, cy=0.5, r=0.2):
    return (X - cx)**2 + (Y - cy)**2 <= r**2


def rho_func(X, Y):
    """Densité : 1 dans l'air, 10 dans le disque."""
    return np.where(in_disk(X, Y), 10.0, 1.0)


def sigma_c_func(rho, T):
    """Scattering proportionnel à la densité."""
    return rho * 2.0


def sigma_a_func(rho, T):
    """Pas d'absorption (milieu diélectrique transparent)."""
    return np.zeros_like(rho)