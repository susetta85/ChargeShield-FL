"""
Test puro-Python (nessuna dipendenza sklearn/torch, entrambe assenti in
questo sandbox — stesso limite di tests/test_fedmia_gradient_pairing.py)
per la formula di MIA Advantage (task #41, Sprint 10zz+13, 2026-09-02).

Replica ESATTA della definizione matematica usata da _mia_advantage() in
scripts/run_experiments.py (Adv = max_t(TPR(t) - FPR(t)), la statistica J
di Youden), non una chiamata alla funzione reale (che importa
sklearn.metrics.roc_curve, non disponibile qui) — verifica quindi la
correttezza della FORMULA, non il percorso di codice sklearn stesso. Se
questa formula cambia in run_experiments.py, aggiornare insieme qui.
"""
import pytest


def _roc_points(labels: list[int], scores: list[float]) -> list[tuple[float, float]]:
    """
    Costruisce i punti (FPR, TPR) della curva ROC per ogni soglia distinta
    presente in `scores`, includendo la soglia +inf (nessun positivo
    predetto: FPR=TPR=0). Implementazione pura-Python indipendente da
    sklearn, usata SOLO per verificare la formula in isolamento.
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    thresholds = sorted(set(scores), reverse=True)
    points = [(0.0, 0.0)]
    for t in thresholds:
        tp = sum(1 for l, s in zip(labels, scores) if l == 1 and s >= t)
        fp = sum(1 for l, s in zip(labels, scores) if l == 0 and s >= t)
        tpr = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        points.append((fpr, tpr))
    return points


def _mia_advantage_reference(labels: list[int], scores: list[float]) -> float:
    """Replica pura-Python di _mia_advantage(): max(TPR - FPR) su ogni soglia."""
    points = _roc_points(labels, scores)
    return max(tpr - fpr for fpr, tpr in points)


def test_perfect_separation_gives_advantage_one():
    # Ogni membro ha score piu' alto di ogni non-membro: un attaccante
    # perfetto esiste, Adv deve raggiungere il massimo teorico 1.0.
    labels = [1, 1, 1, 0, 0, 0]
    scores = [10.0, 9.0, 8.0, 3.0, 2.0, 1.0]
    assert _mia_advantage_reference(labels, scores) == pytest.approx(1.0)


def test_identical_distributions_give_advantage_near_zero():
    # Stessi score per entrambe le classi, alternati: nessuna soglia separa
    # meglio del caso, Adv deve restare vicino a 0 (non necessariamente
    # esattamente 0 per via della discretizzazione a n piccolo).
    labels = [1, 0, 1, 0, 1, 0, 1, 0]
    scores = [5.0, 5.0, 4.0, 4.0, 3.0, 3.0, 2.0, 2.0]
    assert _mia_advantage_reference(labels, scores) == pytest.approx(0.0, abs=1e-9)


def test_advantage_between_zero_and_one_for_partial_separation():
    # Caso intermedio, realistico (score sovrapposti ma con una tendenza):
    # Adv deve cadere in (0, 1) esclusi gli estremi.
    labels = [1, 1, 1, 0, 0, 0]
    scores = [5.0, 3.0, 1.0, 4.0, 2.0, 0.5]
    adv = _mia_advantage_reference(labels, scores)
    assert 0.0 < adv < 1.0


def test_advantage_matches_max_tpr_minus_fpr_at_best_threshold():
    # Verifica diretta contro un calcolo a mano per un caso piccolo e noto:
    # alla soglia t=3.5, TPR=2/3 (membri con score>=3.5: 5,4 -> 2 su 3),
    # FPR=0/3 (nessun non-membro >= 3.5) -> TPR-FPR = 0.667, il massimo
    # raggiungibile in questo set (verificato non essercene uno migliore).
    labels = [1, 1, 1, 0, 0, 0]
    scores = [5.0, 4.0, 1.0, 3.0, 2.0, 0.5]
    adv = _mia_advantage_reference(labels, scores)
    assert adv == pytest.approx(2 / 3)


def test_advantage_is_symmetric_to_score_sign_flip_when_labels_flip():
    # Invertire sia i label sia il segno degli score (== scambiare quale
    # classe l'attaccante chiama "membro") deve dare lo stesso Adv massimo
    # — la statistica J è simmetrica per costruzione (max su ENTRAMBE le
    # direzioni di soglia non e' assunto qui, solo threshold>=t crescenti,
    # quindi verifichiamo lo scenario a separazione perfetta invertito).
    labels = [1, 1, 1, 0, 0, 0]
    scores = [10.0, 9.0, 8.0, 3.0, 2.0, 1.0]
    flipped_labels = [0, 0, 0, 1, 1, 1]
    flipped_scores = [-s for s in scores]
    assert _mia_advantage_reference(flipped_labels, flipped_scores) == pytest.approx(1.0)
