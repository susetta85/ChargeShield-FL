"""
Test puro-Python per il fix del bug di collisione di chiavi tra
"tpr_at_fpr_*" del solo ultimo round e "composed_tpr_at_fpr_*" del composto
multi-round (task #66, Sprint 10zz+41, 2026-09-04).

Contesto del bug (trovato da un audit del codice, non da un run fallito):
run_lira() scriveva `composed_output.update(_tpr_at_fixed_fpr(...))` — chiavi
BARE ("tpr_at_fpr_0.001" ecc.), diverse da ogni altro campo composto nella
stessa funzione ("composed_lira_auc_roc", "composed_lira_advantage",
"composed_lira_confusion" — tutti con prefisso "composed_"). Il merge finale
in src/plugins/attacks/lira.py (`results[_final_round].update(_composed)`)
sovrascriveva quindi silenziosamente il tpr_at_fpr_* del SOLO ultimo round
(calcolato correttamente qualche riga prima in run_lira(), sulla sola
evidenza di quel round) con il valore CUMULATIVO multi-round — un dato
diverso, non un duplicato innocuo. Bug live dal Sprint 10pp (2026-08-28, da
quando TPR@low-FPR esiste) fino ad oggi: ogni run con LiRA (sempre nel
registro di default) aveva l'ultimo round con tpr_at_fpr_* che era in realtà
il valore composto, non quello del round.

_tpr_at_fixed_fpr() stessa non è replicata qui (dipende da sklearn.roc_curve,
non disponibile in questo sandbox, già coperta concettualmente da test
esistenti su TPR@low-FPR) — questo test verifica SOLO la trasformazione di
chiave introdotta dal fix (prefissatura "composed_"), che è pura
manipolazione di dict e quindi testabile in isolamento senza sklearn/torch.
"""


def _apply_composed_prefix_fix(tpr_fields: dict[str, float | None]) -> dict[str, float | None]:
    """Replica esatta della riga corretta in run_lira() (scripts/run_experiments.py):
        for _k, _v in _tpr_at_fixed_fpr(_labels, _scores).items():
            composed_output[f"composed_{_k}"] = _v
    """
    return {f"composed_{k}": v for k, v in tpr_fields.items()}


def test_composed_keys_are_prefixed():
    tpr_fields = {"tpr_at_fpr_0.001": 0.01, "tpr_at_fpr_0.01": 0.05, "tpr_at_fpr_0.05": 0.12}
    composed = _apply_composed_prefix_fix(tpr_fields)
    assert composed == {
        "composed_tpr_at_fpr_0.001": 0.01,
        "composed_tpr_at_fpr_0.01": 0.05,
        "composed_tpr_at_fpr_0.05": 0.12,
    }


def test_composed_keys_never_collide_with_round_own_keys():
    # Il round finale (già presente in `results[_final_round]` PRIMA del
    # merge, popolato da run_lira() con le proprie chiavi bare) non deve
    # perdere nessun campo dopo `.update(_composed)` col fix applicato.
    round_own_fields = {
        "lira_auc_roc": 0.5043,
        "tpr_at_fpr_0.001": 0.0012,
        "tpr_at_fpr_0.01": 0.0123,
        "tpr_at_fpr_0.05": 0.0512,
    }
    composed_fields_raw = {"tpr_at_fpr_0.001": 0.0009, "tpr_at_fpr_0.01": 0.0098, "tpr_at_fpr_0.05": 0.0489}
    composed = _apply_composed_prefix_fix(composed_fields_raw)

    merged = dict(round_own_fields)
    merged.update(composed)

    # Il valore del solo ultimo round resta quello originale, non sovrascritto.
    assert merged["tpr_at_fpr_0.001"] == 0.0012
    assert merged["tpr_at_fpr_0.01"] == 0.0123
    assert merged["tpr_at_fpr_0.05"] == 0.0512
    # Il valore composto resta comunque disponibile, sotto una chiave dedicata.
    assert merged["composed_tpr_at_fpr_0.001"] == 0.0009
    assert merged["composed_tpr_at_fpr_0.01"] == 0.0098
    assert merged["composed_tpr_at_fpr_0.05"] == 0.0489


def test_pre_fix_behavior_would_have_collided():
    # Documenta il bug stesso: la vecchia riga (`composed_output.update(...)`
    # SENZA prefisso) avrebbe sovrascritto silenziosamente i valori del
    # round — verificato qui per contrasto, non perché sia il comportamento
    # corretto.
    round_own_fields = {"tpr_at_fpr_0.01": 0.0123}
    composed_fields_raw = {"tpr_at_fpr_0.01": 0.0098}

    buggy_merged = dict(round_own_fields)
    buggy_merged.update(composed_fields_raw)  # comportamento pre-fix
    assert buggy_merged["tpr_at_fpr_0.01"] == 0.0098  # valore del round PERSO

    fixed_merged = dict(round_own_fields)
    fixed_merged.update(_apply_composed_prefix_fix(composed_fields_raw))
    assert fixed_merged["tpr_at_fpr_0.01"] == 0.0123  # valore del round preservato
    assert fixed_merged["composed_tpr_at_fpr_0.01"] == 0.0098
