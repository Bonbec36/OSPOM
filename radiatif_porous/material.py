"""
material.py — Propriétés optiques du milieu.

Cas test : disque diélectrique centré dans un carré d'air.
  - Air    : sigma_c faible, sigma_a nul
  - Disque : sigma_c fort (diffusion), sigma_a non nul (absorption partielle)
"""

import numpy as np

# ── Géométrie ─────────────────────────────────────────────────────────────────

DISK_CX = 0.5
DISK_CY = 0.5
DISK_R  = 0.3

def in_disk(X, Y):
    return (X - DISK_CX)**2 + (Y - DISK_CY)**2 <= DISK_R**2

# ── Propriétés optiques ───────────────────────────────────────────────────────

RHO_AIR   = 1.0
RHO_DISK  = 10.0

SIGMA_C_AIR  = 0.5    # scattering faible dans l'air
SIGMA_C_DISK = 5.0   # scattering fort dans le diélectrique

SIGMA_A_AIR  = 0.0    # pas d'absorption dans l'air
SIGMA_A_DISK = 2.0    # absorption partielle dans le disque


def rho_func(X, Y):
    return np.where(in_disk(X, Y), RHO_DISK, RHO_AIR)

def sigma_c_func(rho, T):
    """Section efficace de scattering — constante par matériau."""
    return np.where(rho > (RHO_AIR + RHO_DISK) / 2, SIGMA_C_DISK, SIGMA_C_AIR)

def sigma_a_func(rho, T):
    """Section efficace d'absorption — non nulle dans le disque."""
    return np.where(rho > (RHO_AIR + RHO_DISK) / 2, SIGMA_A_DISK, SIGMA_A_AIR)