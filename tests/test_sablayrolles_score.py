"""
Test puro-Python (nessuna dipendenza numpy/torch/sklearn) per la formula
dell'attacco di Sablayrolles et al. 2019 [56] (task #58, Sprint 10zz+33,
2026-09-03) — vedi _sablayrolles_score() in scripts/run_experiments.py e
docs/MetricsReference_DSN2027.md §3.

Stesso limite/pattern di tests/test_mia_advantage.py: run_experiments.py
importa torch a livello di modulo (per Autoencoder), quindi non è
importabile in questo sandbox — questi test replicano la formula ESATTA
(non chiamano la funzione reale) per verificarne la correttezza matematica
in isolamento. _sablayrolles_score() stessa non dipende da numpy/sklearn
(solo aritmetica su float), quindi la replica qui è byte-per-byte identica
alla formula reale, non un'approssimazione.

Formula verificata (Carlini et al. 2022 §V-C, citando Sablayrolles et al.
2019): soglia non-parametrica per-esempio τ_{x,y} = (μ_in+μ_out)/2, con
punteggio = τ_{x,y} - target_loss (convenzione di segno di questo progetto:
più alto = più probabile membro, coerente con lira_score e con l'attacco
LOSS di Yeom -ℓ(x,y) > τ).
"""


import pytest


def _sablayrolles_score_reference(mu_in: float, mu_out: float, target_loss: float) -> float:
    """Replica esatta di _sablayrolles_score() in scripts/run_experiments.py."""
    return float(((mu_in + mu_out) / 2.0) - target_loss)


def test_score_zero_exactly_at_threshold():
    # target_loss esattamente sulla soglia (μ_in+μ_out)/2 -> punteggio 0,
    # il caso limite/decisione neutra.
    assert _sablayrolles_score_reference(mu_in=1.0, mu_out=3.0, target_loss=2.0) == 0.0


def test_loss_matching_mu_in_gives_positive_score():
    # Un campione la cui loss coincide con μ_in (la distribuzione "membro")
    # deve ricevere un punteggio positivo -> predetto membro, stessa
    # direzione di lira_score e dell'attacco LOSS di Yeom.
    score = _sablayrolles_score_reference(mu_in=1.0, mu_out=3.0, target_loss=1.0)
    assert score > 0


def test_loss_matching_mu_out_gives_negative_score():
    # Simmetrico al test precedente: loss coincidente con μ_out (la
    # distribuzione "non membro") -> punteggio negativo.
    score = _sablayrolles_score_reference(mu_in=1.0, mu_out=3.0, target_loss=3.0)
    assert score < 0


def test_score_is_translation_invariant():
    # Proprietà matematica attesa: spostare μ_in, μ_out, target_loss della
    # stessa costante non deve cambiare il punteggio (la soglia è una MEDIA
    # dei due, quindi trasla insieme al bersaglio).
    base = _sablayrolles_score_reference(mu_in=0.5, mu_out=1.5, target_loss=0.9)
    shifted = _sablayrolles_score_reference(mu_in=0.5 + 10.0, mu_out=1.5 + 10.0, target_loss=0.9 + 10.0)
    # pytest.approx: somma/sottrazione di float con una costante grande
    # (10.0) introduce un errore di arrotondamento dell'ordine di 1e-16 sul
    # risultato finale — irrilevante per l'invarianza matematica testata,
    # ma un confronto == esatto tra float lo farebbe fallire spuriamente.
    assert base == pytest.approx(shifted)


def test_score_symmetric_around_midpoint():
    # Due loss equidistanti dalla soglia su lati opposti devono dare
    # punteggi opposti in segno e uguali in valore assoluto.
    mu_in, mu_out = 2.0, 6.0  # soglia = 4.0
    low = _sablayrolles_score_reference(mu_in, mu_out, target_loss=3.0)   # -1 da soglia
    high = _sablayrolles_score_reference(mu_in, mu_out, target_loss=5.0)  # +1 da soglia
    assert low == -high


def test_ranking_matches_membership_on_toy_population():
    # Popolazione giocattolo dove il modello memorizza chiaramente
    # (membri con loss bassa vicina a μ_in, non-membri con loss alta vicina
    # a μ_out) — il punteggio di Sablayrolles deve separare perfettamente le
    # due classi (AUC=1.0 in questo caso pulito), replicando l'osservazione
    # di Carlini et al. che l'attacco, pur non-parametrico, è efficace.
    mu_in, mu_out = 0.1, 0.5
    member_losses = [0.08, 0.09, 0.11, 0.12, 0.10]
    nonmember_losses = [0.48, 0.52, 0.55, 0.49, 0.51]

    member_scores = [_sablayrolles_score_reference(mu_in, mu_out, t) for t in member_losses]
    nonmember_scores = [_sablayrolles_score_reference(mu_in, mu_out, t) for t in nonmember_losses]

    # Ogni punteggio membro deve essere strettamente maggiore di ogni
    # punteggio non-membro -> separazione perfetta (equivalente ad AUC=1.0).
    assert min(member_scores) > max(nonmember_scores)


def test_score_matches_paper_formula_negated():
    # Verifica esplicita che la nostra convenzione (τ - loss) sia il
    # NEGATIVO della formula letterale del paper A'(x,y) = loss - τ (Carlini
    # et al. 2022 §V-C, citando Sablayrolles et al. [56]) — la negazione è
    # intenzionale per allinearsi alla convenzione di segno "punteggio alto
    # = membro" già usata ovunque in questo file (vedi docstring di
    # _sablayrolles_score() in scripts/run_experiments.py).
    mu_in, mu_out, target_loss = 1.2, 2.8, 1.9
    tau = (mu_in + mu_out) / 2.0
    paper_formula = target_loss - tau
    our_score = _sablayrolles_score_reference(mu_in, mu_out, target_loss)
    assert our_score == -paper_formula
