"""
Test puro-Python (nessuna dipendenza sklearn/torch, entrambe assenti in
questo sandbox — stesso limite di tests/test_mia_advantage.py) per la
formula di MIA Confusion Matrix alla soglia ottimale (task #49, Sprint
10zz+25, 2026-09-03).

Replica ESATTA della definizione matematica usata da
_mia_confusion_at_best_threshold() in scripts/run_experiments.py: trova la
soglia t* che massimizza TPR(t)-FPR(t) (stessa soglia di _mia_advantage()),
poi conta TP/FP/TN/FN classificando come "membro" ogni campione con
score >= t* — non una chiamata alla funzione reale (che importa
sklearn.metrics.roc_curve, non disponibile qui), verifica quindi la
correttezza della FORMULA, non il percorso di codice sklearn stesso. Se
questa formula cambia in run_experiments.py, aggiornare insieme qui.
"""
import pytest


def _roc_points_with_thresholds(
    labels: list[int], scores: list[float]
) -> list[tuple[float, float, float]]:
    """
    Costruisce i punti (threshold, FPR, TPR) della curva ROC per ogni
    soglia distinta presente in `scores`, includendo la soglia +inf
    (nessun positivo predetto: FPR=TPR=0) — stessa logica di
    tests/test_mia_advantage.py::_roc_points, con la soglia esplicitata.
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    thresholds = sorted(set(scores), reverse=True)
    points = [(float("inf"), 0.0, 0.0)]
    for t in thresholds:
        tp = sum(1 for l, s in zip(labels, scores) if l == 1 and s >= t)
        fp = sum(1 for l, s in zip(labels, scores) if l == 0 and s >= t)
        tpr = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        points.append((t, fpr, tpr))
    return points


def _mia_confusion_reference(labels: list[int], scores: list[float]) -> dict:
    """Replica pura-Python di _mia_confusion_at_best_threshold()."""
    points = _roc_points_with_thresholds(labels, scores)
    best_threshold, best_fpr, best_tpr = max(points, key=lambda p: p[2] - p[1])
    advantage = best_tpr - best_fpr

    n_members = sum(labels)
    n_nonmembers = len(labels) - n_members
    tp = sum(1 for l, s in zip(labels, scores) if l == 1 and s >= best_threshold)
    fp = sum(1 for l, s in zip(labels, scores) if l == 0 and s >= best_threshold)
    fn = n_members - tp
    tn = n_nonmembers - fp

    return {
        "threshold": best_threshold, "advantage": advantage,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_members": n_members, "n_nonmembers": n_nonmembers,
    }


def test_perfect_separation_gives_all_members_detected_zero_false_positives():
    # Ogni membro ha score piu' alto di ogni non-membro: alla soglia
    # ottimale l'attaccante rileva TUTTI i membri (tp=n_members, fn=0)
    # senza falsi allarmi (fp=0, tn=n_nonmembers).
    labels = [1, 1, 1, 0, 0, 0]
    scores = [10.0, 9.0, 8.0, 3.0, 2.0, 1.0]
    result = _mia_confusion_reference(labels, scores)
    assert result["tp"] == 3
    assert result["fn"] == 0
    assert result["fp"] == 0
    assert result["tn"] == 3
    assert result["n_members"] == 3
    assert result["n_nonmembers"] == 3


def test_confusion_counts_sum_to_class_totals():
    # Invariante strutturale, indipendente dal caso specifico: tp+fn deve
    # sempre dare il numero totale di membri, fp+tn quello dei non-membri
    # — vale per qualunque soglia, non solo quella ottimale.
    labels = [1, 1, 1, 0, 0, 0, 0]
    scores = [5.0, 3.0, 1.0, 4.0, 2.0, 0.5, 4.5]
    result = _mia_confusion_reference(labels, scores)
    assert result["tp"] + result["fn"] == result["n_members"] == 3
    assert result["fp"] + result["tn"] == result["n_nonmembers"] == 4


def test_confusion_matches_known_best_threshold():
    # Stesso caso di test_advantage_matches_max_tpr_minus_fpr_at_best_threshold
    # in test_mia_advantage.py: soglia ottimale t=3.5, membri con score>=3.5
    # sono {5.0, 4.0} -> tp=2, fn=1 (il membro con score=1.0 sfugge);
    # nessun non-membro >= 3.5 -> fp=0, tn=3.
    labels = [1, 1, 1, 0, 0, 0]
    scores = [5.0, 4.0, 1.0, 3.0, 2.0, 0.5]
    result = _mia_confusion_reference(labels, scores)
    assert result["tp"] == 2
    assert result["fn"] == 1
    assert result["fp"] == 0
    assert result["tn"] == 3
    assert result["advantage"] == pytest.approx(2 / 3)


def test_confusion_advantage_matches_mia_advantage_formula():
    # L'advantage riportato da _mia_confusion_at_best_threshold() deve
    # coincidere esattamente con quello di _mia_advantage() sullo stesso
    # input — stessa soglia, stessa formula, due funzioni indipendenti che
    # non devono divergere.
    labels = [1, 1, 1, 0, 0, 0]
    scores = [7.0, 3.0, 2.0, 6.0, 4.0, 1.0]
    result = _mia_confusion_reference(labels, scores)
    points = _roc_points_with_thresholds(labels, scores)
    expected_advantage = max(tpr - fpr for _, fpr, tpr in points)
    assert result["advantage"] == pytest.approx(expected_advantage)


def test_identical_distributions_give_near_chance_confusion():
    # Nessuna soglia separa meglio del caso: alla soglia ottimale (comunque
    # trovata, per costruzione del caso simmetrico) i conteggi non devono
    # mostrare una separazione netta — tp non deve coprire tutti i membri
    # senza anche produrre falsi positivi comparabili.
    labels = [1, 0, 1, 0, 1, 0, 1, 0]
    scores = [5.0, 5.0, 4.0, 4.0, 3.0, 3.0, 2.0, 2.0]
    result = _mia_confusion_reference(labels, scores)
    assert result["advantage"] == pytest.approx(0.0, abs=1e-9)
    # Con Adv=0, TPR=FPR alla soglia scelta: la frazione di membri rilevati
    # deve uguagliare la frazione di non-membri erroneamente rilevati.
    tpr = result["tp"] / result["n_members"]
    fpr = result["fp"] / result["n_nonmembers"]
    assert tpr == pytest.approx(fpr)
