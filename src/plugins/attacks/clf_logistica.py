"""Classificatore di tipo Shokri per lo shadow (shadow.classificatore: logistica).

Regressione logistica addestrata sugli shadow: input = errore di ricostruzione per
feature di ogni sessione (un vettore per sessione), etichetta 1 = membro dello shadow,
0 = non membro (calibrazione). Pesi di classe bilanciati, standardizzazione con le sole
statistiche dei dati di addestramento dello shadow, discesa del gradiente deterministica.
Solo numpy: nessuna dipendenza da torch, quindi testabile da sola."""
from __future__ import annotations

import numpy as np


def regressione_logistica(x1, x0, passi: int = 2000, passo: float = 0.1, l2: float = 1e-3) -> dict:
    x1 = np.asarray(x1, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    if x1.ndim != 2 or x0.ndim != 2 or x1.shape[1] != x0.shape[1]:
        raise ValueError("regressione_logistica: attese due matrici (n, d) con d uguale")
    if len(x1) == 0 or len(x0) == 0:
        raise ValueError("regressione_logistica: insieme 1 o 0 vuoto")
    x = np.vstack([x1, x0])
    y = np.r_[np.ones(len(x1)), np.zeros(len(x0))]
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    z = (x - mu) / sd
    n = len(y)
    # pesi di classe bilanciati: ogni classe pesa 1/2 in media
    sw = np.where(y == 1, 0.5 / len(x1), 0.5 / len(x0)) * n
    w = np.zeros(z.shape[1])
    b = 0.0
    for _ in range(passi):
        s = z @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(s, -500, 500)))
        g = sw * (p - y)
        w = w - passo * (z.T @ g / n + l2 * w)
        b = b - passo * float(g.mean())
    return {"mu": mu, "sd": sd, "w": w, "b": float(b)}


def punteggio_logistica(modello: dict, x) -> np.ndarray:
    """Logit per ogni riga: >= 0 significa 'membro' (probabilita' >= 1/2)."""
    x = np.asarray(x, dtype=float)
    return ((x - modello["mu"]) / modello["sd"]) @ modello["w"] + modello["b"]
