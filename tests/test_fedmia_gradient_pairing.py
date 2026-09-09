# tests/test_fedmia_gradient_pairing.py
"""
Test non-torch della logica di pairing/skip in run_fedmia_gradient()
(scripts/run_experiments.py) — replica ESATTA (non un'approssimazione) delle
righe che decidono train/test split e skip, estratta come funzione pura
testabile senza torch installato.

Motivo: run_fedmia_gradient() nel suo complesso richiede torch (Autoencoder,
AutoencoderTrainer, GradientManager) e non può essere eseguito in un ambiente
senza torch — ma la logica di pairing/skip qui sotto (quali shadow finiscono
in train vs test, quando saltare un cluster/round) è pura logica di liste,
indipendente da torch, e QUESTA parte può e deve essere verificata qui.

Storia (2026-09-01, entrambe le versioni trovate SOLO da un run reale, non
per ispezione del codice):

1. Trovato da una review indipendente, PRIMA di qualunque esecuzione reale:
   con n_shadow piccolo (default 8), train_member poteva avere lunghezza 1,
   che FedMIA.calibrate_from_vectors() (src/plugins/attacks/fedmia.py)
   rifiutava con un crash di nn.BatchNorm1d ("Expected more than 1 value per
   channel"). Fix v1: skip esplicito quando len(train_member) < 2.

2. Trovato da un run reale (smoke test --n-shadow 16, 2026-09-01 14:35):
   il fix v1 evitava il crash, ma con lo split 0::2/1::2 sull'elenco
   COMBINATO (non per classe), il test set finiva spesso con UNA SOLA
   classe — n_test=24 (tutti e 3 i cluster ammessi) ma auc=None in
   ENTRAMBI i round, perché l'universo membri:non-membri è sbilanciato
   ~4:1 (split 80/20) e i pochi shadow "a maggioranza non-membro" cadevano
   per caso tutti in posizione pari o dispari. Fix v2 (quello testato qui):
   split STRATIFICATO PER CLASSE — membri e non-membri divisi
   separatamente in train/test, garantendo entrambe le classi nel test set
   ogni volta che ce ne sono almeno 2 per classe.
"""

from __future__ import annotations

import pytest


def _pairing_logic(
    vectors: list[object | None],
    is_member_majority: list[bool],
) -> str | tuple[str, int, int, int, int]:
    """
    Replica ESATTA di scripts/run_experiments.py::run_fedmia_gradient() (fix
    v2, 2026-09-01, split stratificato per classe). Se questa logica cambia
    in run_experiments.py, aggiornare anche qui — non importare da
    run_experiments.py perché quel modulo richiede torch al top-level.
    """
    paired = [(v, m) for v, m in zip(vectors, is_member_majority) if v is not None]
    if len(paired) < 4:
        return "SKIP_TOO_FEW_SHADOW"

    members = [v for v, m in paired if m]
    non_members = [v for v, m in paired if not m]
    train_member = members[0::2]
    test_member = members[1::2]
    train_non_member = non_members[0::2]
    test_non_member = non_members[1::2]

    if len(train_member) < 2 or not test_member or not test_non_member:
        return "SKIP_TOO_FEW_TRAIN_MEMBER"

    return (
        "OK",
        len(train_member),
        len(train_non_member),
        len(test_member),
        len(test_non_member),
    )


@pytest.mark.parametrize(
    "name,vectors,majority",
    [
        (
            "n_shadow=8, alternata M/NM",
            ["v"] * 8,
            [True, False, True, False, True, False, True, False],
        ),
        (
            "n_shadow=8, quasi tutti membri (7M/1NM)",
            ["v"] * 8,
            [True, True, True, True, True, True, True, False],
        ),
        (
            "n_shadow=8, quasi tutti non-membri (1M/7NM)",
            ["v"] * 8,
            [True, False, False, False, False, False, False, False],
        ),
        ("n_shadow=8, tutti membri", ["v"] * 8, [True] * 8),
        ("n_shadow=8, tutti non-membri", ["v"] * 8, [False] * 8),
        ("n_shadow=16, alternata (paper quality)", ["v"] * 16, [True, False] * 8),
        ("n_shadow=32, alternata (paper quality alto)", ["v"] * 32, [True, False] * 16),
        (
            "n_shadow=8 con 2 shadow skippati (vettore None)",
            ["v", None, "v", "v", None, "v", "v", "v"],
            [True, True, False, True, True, False, True, False],
        ),
        ("n_shadow=3 (troppo pochi shadow totali)", ["v"] * 3, [True, False, True]),
        ("n_shadow=0 (cluster vuoto)", [], []),
        (
            "n_shadow=16, sbilanciato 4:1 con non-membri consecutivi "
            "(caso reale del run 2026-09-01 14:35 — prima del fix v2 dava "
            "auc=None nonostante n_test=24)",
            ["v"] * 16,
            [True] * 13 + [False] * 3,
        ),
        (
            "n_shadow=16, l'unico non-membro cade in posizione pari "
            "(finirebbe SOLO in train con split non stratificato — "
            "esattamente il bug del fix v1/pre-v2)",
            ["v"] * 16,
            [True, True, False] + [True] * 13,
        ),
    ],
)
def test_pairing_logic_never_raises(name, vectors, majority):
    """Nessuna composizione degenere deve sollevare un'eccezione — solo SKIP
    esplicito."""
    result = _pairing_logic(vectors, majority)
    assert result == "SKIP_TOO_FEW_SHADOW" or result == "SKIP_TOO_FEW_TRAIN_MEMBER" or (
        isinstance(result, tuple) and result[0] == "OK"
    )


def test_pairing_logic_ok_case_has_at_least_two_train_member():
    """Quando il risultato è OK, train_member deve avere sempre >= 2 elementi
    (il vincolo che FedMIA.calibrate_from_vectors() richiede per non far
    crashare nn.BatchNorm1d con batch_size=1)."""
    result = _pairing_logic(
        ["v"] * 8, [True, False, True, False, True, False, True, False]
    )
    assert result[0] == "OK"
    assert result[1] >= 2  # len(train_member)


def test_pairing_logic_ok_case_test_set_has_both_classes():
    """Quando il risultato è OK, il test set deve SEMPRE contenere entrambe
    le classi (fix v2, 2026-09-01) — altrimenti roc_auc_score() in
    run_fedmia_gradient() darebbe auc=None nonostante n_test>0, come
    osservato nel run reale 2026-09-01 14:35 (n_test=24, auc=None in
    entrambi i round) prima di questo fix."""
    result = _pairing_logic(
        ["v"] * 16, [True] * 13 + [False] * 3
    )
    assert result[0] == "OK"
    n_test_member, n_test_non_member = result[3], result[4]
    assert n_test_member > 0
    assert n_test_non_member > 0


def test_pairing_logic_skips_when_all_shadows_same_class():
    """Se tutti gli shadow di un cluster/round sono a maggioranza non-membro,
    train_member è vuoto (0 < 2) — deve saltare, non crashare."""
    assert _pairing_logic(["v"] * 8, [False] * 8) == "SKIP_TOO_FEW_TRAIN_MEMBER"


def test_pairing_logic_skips_when_only_one_non_member_present():
    """Con un solo shadow non-membro totale, non può finire sia in train sia
    in test — skip esplicito (test_non_member resta vuoto), non un test set
    a una sola classe silenzioso."""
    result = _pairing_logic(["v"] * 8, [True] * 7 + [False] * 1)
    assert result == "SKIP_TOO_FEW_TRAIN_MEMBER"


def test_pairing_logic_skips_when_too_few_valid_shadows():
    """Meno di 4 shadow validi (dopo aver scartato i None) — troppo pochi per
    uno split train/test onesto, skip esplicito."""
    assert _pairing_logic(["v", None, None, None], [True, True, True, True]) == (
        "SKIP_TOO_FEW_SHADOW"
    )


# ── controlled_composition (Sprint 10zz+5, 2026-09-01) ──────────────────────
#
# Terzo problema reale trovato SOLO da un'esecuzione reale (non dalla review
# statica, non dal primo run — dal secondo, con lo split stratificato già
# corretto): con l'universo storico membri:non-membri ~80:20 e uno shadow
# che campiona metà universo, la composizione converge quasi deterministicamente
# vicino a 80:20 — uno shadow "a maggioranza non-membro" è essenzialmente
# impossibile per cluster grandi, a QUALUNQUE n_shadow (osservato: n_test=0 in
# entrambi i round con n_shadow=16). Fix: run_lira() guadagna un parametro
# opt-in controlled_composition che assegna a ogni shadow una frazione-membri
# BERSAGLIO, distribuita uniformemente su [0.1, 0.9] in base al suo indice —
# non più lasciata al caso. Questa sezione testa quella formula pura
# (nessun bisogno di torch: è aritmetica, non training).

def _target_member_fraction(shadow_idx: int, n_shadow: int) -> float:
    """Replica ESATTA della formula in run_lira() (Step 2, ramo
    controlled_composition=True). Aggiornare insieme se cambia là."""
    return 0.1 + 0.8 * (shadow_idx / max(1, n_shadow - 1))


@pytest.mark.parametrize("n_shadow", [2, 4, 8, 16, 32, 64])
def test_controlled_composition_spans_both_classes(n_shadow):
    """Per ogni n_shadow ragionevole, la formula deve produrre ALMENO uno
    shadow con frazione target < 0.5 (candidato non-membro-maggioranza) e
    ALMENO uno con frazione target >= 0.5 (candidato membro-maggioranza) —
    altrimenti il problema osservato nel run reale (n_test=0) si
    ripresenterebbe anche col fix."""
    fractions = [_target_member_fraction(i, n_shadow) for i in range(n_shadow)]
    assert any(f < 0.5 for f in fractions), f"nessuna frazione < 0.5 per n_shadow={n_shadow}"
    assert any(f >= 0.5 for f in fractions), f"nessuna frazione >= 0.5 per n_shadow={n_shadow}"


@pytest.mark.parametrize("n_shadow", [2, 4, 8, 16, 32, 64])
def test_controlled_composition_stays_within_bounds(n_shadow):
    """Nessuna frazione target deve uscire da [0.1, 0.9] — ai due estremi la
    composizione resta comunque mista (mai 100%/0% di una classe), coerente
    col fatto che comunque si campiona da un pool reale."""
    fractions = [_target_member_fraction(i, n_shadow) for i in range(n_shadow)]
    assert all(0.1 <= f <= 0.9 for f in fractions)


def test_controlled_composition_n_shadow_one_does_not_divide_by_zero():
    """n_shadow=1 non deve sollevare ZeroDivisionError (max(1, n_shadow-1)
    nella formula)."""
    assert _target_member_fraction(0, 1) == 0.1
