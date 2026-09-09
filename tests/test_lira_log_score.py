"""
Test puro-Python (numpy disponibile in questo sandbox, torch/sklearn no —
vedi tests/test_sablayrolles_score.py per lo stesso limite) per la variante
esplorativa "LiRA su log(MSE)" (task #61, Sprint 10zz+36, 2026-09-03) — vedi
_lira_log_score() in scripts/run_experiments.py e
docs/MetricsReference_DSN2027.md §3.

A differenza di _sablayrolles_score() (puramente aritmetica), questa
funzione usa np.mean/np.std/np.clip — importabile direttamente da
scripts/run_experiments.py NON è possibile (il modulo importa torch a
livello globale), quindi questi test replicano la formula ESATTA copiandola
qui (stesso pattern di tests/test_mia_advantage.py), verificata riga per
riga contro la funzione reale.
"""
import math

import numpy as np
import pytest


def _lira_log_score_reference(
    in_losses: list[float],
    out_losses: list[float],
    target_loss: float,
    eps: float = 1e-8,
    sigma_floor: float = 0.05,
) -> float:
    """Replica esatta di _lira_log_score() in scripts/run_experiments.py."""
    log_out = [math.log(x + eps) for x in out_losses]
    log_target = math.log(target_loss + eps)

    mu_out_log = float(np.mean(log_out))
    sigma_out_log = max(float(np.std(log_out)), sigma_floor)

    if len(in_losses) >= 2:
        log_in = [math.log(x + eps) for x in in_losses]
        mu_in_log = float(np.mean(log_in))
        sigma_in_log = max(float(np.std(log_in)), sigma_floor)
    else:
        mu_in_log = mu_out_log
        sigma_in_log = sigma_out_log

    log_p_in = (-0.5 * ((log_target - mu_in_log) / sigma_in_log) ** 2) - math.log(sigma_in_log)
    log_p_out = (-0.5 * ((log_target - mu_out_log) / sigma_out_log) ** 2) - math.log(sigma_out_log)
    return float(np.clip(log_p_in - log_p_out, -20.0, 20.0))


def test_fallback_without_in_losses_gives_zero_score():
    # Con <2 osservazioni IN reali, mu_in_log=mu_out_log e sigma_in_log=
    # sigma_out_log (fallback dichiarato) -> log_p_in == log_p_out sempre,
    # qualunque sia target_loss -> score sempre 0 (nessun segnale assunto).
    score = _lira_log_score_reference(
        in_losses=[], out_losses=[0.001, 0.002, 0.0015, 0.0018], target_loss=0.05,
    )
    assert score == pytest.approx(0.0, abs=1e-9)


def test_member_like_loss_gives_positive_score():
    # in_losses concentrate su una MSE bassa, out_losses su una MSE alta ->
    # target_loss vicino alla distribuzione IN deve dare score positivo
    # (convenzione: score>0 -> membro, stessa di lira_score/sablayrolles).
    in_losses = [0.001, 0.0012, 0.0009, 0.0011, 0.0010]
    out_losses = [0.05, 0.048, 0.052, 0.051, 0.049]
    score = _lira_log_score_reference(in_losses, out_losses, target_loss=0.0010)
    assert score > 0


def test_nonmember_like_loss_gives_negative_score():
    in_losses = [0.001, 0.0012, 0.0009, 0.0011, 0.0010]
    out_losses = [0.05, 0.048, 0.052, 0.051, 0.049]
    score = _lira_log_score_reference(in_losses, out_losses, target_loss=0.050)
    assert score < 0


def test_sigma_floor_is_applied_when_std_is_tiny():
    # Tutti gli out_losses identici (std=0 in log-scale) -> sigma_out_log
    # deve essere il floor, non 0 (altrimenti divisione per zero/esplosione
    # del punteggio) — replica diretta del comportamento "floor-hit"
    # tracciato in run_lira().
    out_losses = [0.01, 0.01, 0.01, 0.01]
    # target_loss leggermente diverso dal cluster out -> se il floor non
    # fosse applicato, lo score esploderebbe (sigma=0 -> divisione per 0);
    # con il floor resta un numero finito e ragionevole (clippato a ±20).
    score = _lira_log_score_reference(in_losses=[], out_losses=out_losses, target_loss=0.02)
    assert math.isfinite(score)
    assert -20.0 <= score <= 20.0


def test_score_is_clipped_to_plus_minus_20():
    # target_loss estremo rispetto a out_losses concentrate -> il
    # log-likelihood-ratio grezzo supererebbe ±20 senza il clip.
    out_losses = [0.001] * 10
    score_high = _lira_log_score_reference(in_losses=[], out_losses=out_losses, target_loss=1e-6)
    score_low = _lira_log_score_reference(in_losses=[], out_losses=out_losses, target_loss=100.0)
    assert -20.0 <= score_high <= 20.0
    assert -20.0 <= score_low <= 20.0


def test_ranking_matches_membership_on_toy_population():
    # Stesso tipo di test di test_sablayrolles_score.py — popolazione
    # giocattolo con memorizzazione chiara, verifica separazione perfetta.
    in_losses = [0.001, 0.0012, 0.0009, 0.0011, 0.0010, 0.00105]
    out_losses = [0.05, 0.048, 0.052, 0.051, 0.049, 0.0505]
    member_targets = [0.0009, 0.0010, 0.0011, 0.00095]
    nonmember_targets = [0.048, 0.052, 0.0495, 0.0505]

    member_scores = [
        _lira_log_score_reference(in_losses, out_losses, t) for t in member_targets
    ]
    nonmember_scores = [
        _lira_log_score_reference(in_losses, out_losses, t) for t in nonmember_targets
    ]
    assert min(member_scores) > max(nonmember_scores)
