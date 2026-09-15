"""
Test reali (non repliche) per scripts/check_significance.py — task #70/#67,
Sprint 10zz+42 (2026-09-04). A differenza della maggior parte di questo
progetto, check_significance.py NON importa torch (solo stdlib + scipy
opzionale) — quindi è importabile direttamente in questo sandbox, ed è
possibile testare le funzioni reali invece di repliche pure-Python.

Motivazione: un audit del codice ha trovato che questo script — che calcola
i numeri di significatività statistica citati nel paper — non aveva NESSUN
test, nonostante abbia già una storia di due bug reali di conflazione
silenziosa trovati e corretti nel suo stesso `discover_groups()` (mapping
cartella→epsilon, 2026-08-27; cartelle diagnostiche non escluse, 2026-08-28).
Questo file copre in particolare la terza conflazione trovata oggi
(`entity-split-sweep1` raggruppato silenziosamente con `nodp-sweep1` sotto
"no-DP baseline", n=10 invece di n=5 — vedi commento su
`_METHODOLOGY_VARIANT_PREFIXES` in check_significance.py) come test di
regressione, così non può ripresentarsi inosservato.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_significance as cs  # noqa: E402


def _write_experiment(
    path: Path,
    *,
    dp_mode: str = "dp-fedavg",
    epsilon: float | None = None,
    no_dp: bool = True,
    mean_lira_auc_roc: float = 0.5,
    seed: int | None = None,
) -> None:
    # `seed` di default None: la maggior parte dei test qui sotto verifica il
    # raggruppamento per config, non la deduplicazione per seed (task #73,
    # Sprint 10zz+46) — quei test devono passare `seed` esplicito e DIVERSO
    # per ogni file scritto nello stesso gruppo, altrimenti collidono tutti
    # sotto la stessa chiave (label, seed) e discover_groups() ne tiene solo
    # uno (comportamento corretto, ma non quello che quei test vogliono
    # esercitare).
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "config": {
            "dp_mode": dp_mode, "epsilon": epsilon, "no_dp": no_dp, "delta": 1e-5,
            "seed": seed,
        },
        "summary": {"mean_lira_auc_roc": mean_lira_auc_roc},
    }))


class TestDiscoverGroups:
    def test_groups_by_config_content_not_folder_name(self, tmp_path):
        # Due file nella STESSA cartella con config diversi finiscono in
        # gruppi diversi — dimostra che il raggruppamento legge il config,
        # non il nome della cartella (il fix originale del 2026-08-27).
        _write_experiment(tmp_path / "some-sweep" / "experiment_1.json",
                           dp_mode="central", epsilon=1.0, no_dp=False)
        _write_experiment(tmp_path / "some-sweep" / "experiment_2.json",
                           dp_mode="central", epsilon=0.1, no_dp=False)
        groups = cs.discover_groups(pattern=str(tmp_path / "*/experiment_*.json"))
        assert set(groups.keys()) == {"central, eps=1.0", "central, eps=0.1"}

    def test_diagnostic_dirs_excluded_by_default(self, tmp_path):
        _write_experiment(tmp_path / "nodp-sweep1" / "experiment_1.json", seed=42)
        _write_experiment(tmp_path / "_calibration_overfit" / "experiment_1.json", seed=99)
        groups = cs.discover_groups(pattern=str(tmp_path / "*/experiment_*.json"))
        assert sum(len(v) for v in groups.values()) == 1
        groups_all = cs.discover_groups(
            pattern=str(tmp_path / "*/experiment_*.json"), include_diagnostic=True
        )
        assert sum(len(v) for v in groups_all.values()) == 2

    def test_entity_split_excluded_from_no_dp_baseline_by_default(self, tmp_path):
        # Regression test per il bug trovato oggi (2026-09-04): stesso
        # config (no_dp=True) in due cartelle metodologicamente diverse.
        # Seed distinti tra i due gruppi di file (0-4 vs 100-104): altrimenti
        # collidono con la deduplicazione per seed del task #73 e questo test
        # non eserciterebbe più la logica che vuole verificare.
        for seed in range(5):
            _write_experiment(tmp_path / "nodp-sweep1" / f"experiment_{seed}.json", seed=seed)
        for seed in range(5):
            _write_experiment(
                tmp_path / "entity-split-sweep1" / f"experiment_{seed}.json", seed=100 + seed
            )

        groups = cs.discover_groups(pattern=str(tmp_path / "*/experiment_*.json"))
        assert len(groups["no-DP baseline"]) == 5  # NON 10 — entity-split escluso

        groups_with_variants = cs.discover_groups(
            pattern=str(tmp_path / "*/experiment_*.json"),
            include_methodology_variants=True,
        )
        assert len(groups_with_variants["no-DP baseline"]) == 10  # reintegrato esplicitamente

    def test_no_dp_and_epsilon_none_both_map_to_baseline_label(self, tmp_path):
        _write_experiment(tmp_path / "a" / "experiment_1.json", no_dp=True, epsilon=None, seed=1)
        _write_experiment(tmp_path / "b" / "experiment_1.json", no_dp=False, epsilon=None, seed=2)
        groups = cs.discover_groups(pattern=str(tmp_path / "*/experiment_*.json"))
        assert list(groups.keys()) == ["no-DP baseline"]
        assert len(groups["no-DP baseline"]) == 2

    def test_malformed_json_skipped_not_raised(self, tmp_path):
        good = tmp_path / "sweep" / "experiment_1.json"
        good.parent.mkdir(parents=True, exist_ok=True)
        good.write_text("{not valid json")
        _write_experiment(tmp_path / "sweep" / "experiment_2.json")
        groups = cs.discover_groups(pattern=str(tmp_path / "*/experiment_*.json"))
        assert sum(len(v) for v in groups.values()) == 1

    def test_rerun_of_same_seed_deduplicated_keeps_latest(self, tmp_path):
        # Regression test per il bug del task #52/#73 (2026-09-08): OGNI
        # config della campagna e' stata rilanciata con gli stessi 5 seed per
        # aggiungere metriche mancanti (Advantage/Confusion/TPR/Sablayrolles/
        # log-LiRA) — non per ottenere nuova potenza statistica. Prima di
        # questo fix, un seed rieseguito 2-3 volte contava 2-3 volte in
        # discover_groups(), gonfiando n. Qui: stesso seed, due timestamp
        # diversi (nomi file ordinabili cronologicamente) -> deve restare
        # UN SOLO file, quello piu' recente.
        _write_experiment(
            tmp_path / "central-sweep1" / "experiment_20260829_000333.json",
            dp_mode="central", epsilon=1.0, no_dp=False, seed=42, mean_lira_auc_roc=0.5009,
        )
        _write_experiment(
            tmp_path / "central-sweep3" / "experiment_20260902_222826.json",
            dp_mode="central", epsilon=1.0, no_dp=False, seed=42, mean_lira_auc_roc=0.4992,
        )
        groups = cs.discover_groups(pattern=str(tmp_path / "*/experiment_*.json"))
        files = groups["central, eps=1.0"]
        assert len(files) == 1
        assert "20260902_222826" in files[0]  # il PIU' recente, non il primo

    def test_rerun_deduplication_is_per_seed_not_per_group(self, tmp_path):
        # 5 seed, ognuno rieseguito 2 volte nello stesso gruppo -> deve
        # restare esattamente 1 file PER SEED (5 totali), non 1 file per
        # l'intero gruppo.
        for seed in (42, 123, 456, 789, 1234):
            _write_experiment(
                tmp_path / "dp-sweep1" / f"experiment_old_{seed}.json",
                dp_mode="dp-fedavg", epsilon=1.0, no_dp=False, seed=seed,
            )
            _write_experiment(
                tmp_path / "dp-sweep4" / f"experiment_zzz_new_{seed}.json",
                dp_mode="dp-fedavg", epsilon=1.0, no_dp=False, seed=seed,
            )
        groups = cs.discover_groups(pattern=str(tmp_path / "*/experiment_*.json"))
        files = groups["dp-fedavg, eps=1.0"]
        assert len(files) == 5
        assert all("zzz_new" in f for f in files)  # sempre il piu' recente per ogni seed

    def test_no_seed_field_treated_as_single_shared_key(self, tmp_path):
        # Comportamento esplicito, non un edge case ignorato: se il config
        # non ha "seed" (chiave assente, non solo None), tutti i file con lo
        # stesso config finiscono sotto la stessa chiave (label, None) e solo
        # l'ultimo sopravvive — coerente con "un solo valore per combinazione
        # config+seed", dove seed=assente e' esso stesso un valore.
        _write_experiment(tmp_path / "a" / "experiment_1.json")
        _write_experiment(tmp_path / "b" / "experiment_2.json")
        groups = cs.discover_groups(pattern=str(tmp_path / "*/experiment_*.json"))
        assert sum(len(v) for v in groups.values()) == 1

    def test_seed_type_mismatch_still_deduplicated(self, tmp_path):
        # Caso limite non coperto dai test sopra: lo stesso seed "logico" (42)
        # salvato con tipi diversi nel JSON — int 42 in un run, stringa "42"
        # in un rerun (es. valore quotato in un experiment.yaml modificato a
        # mano, o una futura versione dello script che serializza diversamente).
        # Senza normalizzazione del tipo, (label, 42) e (label, "42") sono
        # chiavi PYTHON DIVERSE (42 != "42", anche se hash(42) == hash("42")
        # non vale in Python) -> discover_groups() le tratterebbe come due
        # seed distinti, silenziosamente gonfiando n esattamente come il bug
        # del task #52/#73 che questo stesso fix doveva chiudere.
        json_int = json.dumps({
            "config": {"dp_mode": "dp-fedavg", "epsilon": 1.0, "no_dp": False,
                       "delta": 1e-5, "seed": 42},
            "summary": {"mean_lira_auc_roc": 0.51},
        })
        json_str = json.dumps({
            "config": {"dp_mode": "dp-fedavg", "epsilon": 1.0, "no_dp": False,
                       "delta": 1e-5, "seed": "42"},
            "summary": {"mean_lira_auc_roc": 0.52},
        })
        p_int = tmp_path / "dp-sweep1" / "experiment_20260101_000000.json"
        p_str = tmp_path / "dp-sweep2" / "experiment_20260102_000000.json"
        p_int.parent.mkdir(parents=True, exist_ok=True)
        p_str.parent.mkdir(parents=True, exist_ok=True)
        p_int.write_text(json_int)
        p_str.write_text(json_str)

        groups = cs.discover_groups(pattern=str(tmp_path / "*/experiment_*.json"))
        files = groups["dp-fedavg, eps=1.0"]
        assert len(files) == 1  # non 2 - stesso seed logico, tipo diverso
        assert "20260102_000000" in files[0]  # tiene il più recente

    def test_none_seed_not_conflated_with_string_seed(self, tmp_path):
        # Contro-prova: seed assente (None) e un seed reale (anche se
        # rappresentato come stringa) NON devono mai collassare sotto la
        # stessa chiave solo perche' la normalizzazione del test sopra
        # converte i seed non-None in stringa.
        _write_experiment(tmp_path / "a" / "experiment_1.json", seed=None)
        _write_experiment(tmp_path / "b" / "experiment_2.json", seed=7)
        groups = cs.discover_groups(pattern=str(tmp_path / "*/experiment_*.json"))
        assert sum(len(v) for v in groups.values()) == 2


class TestBootstrapCI:
    def test_none_for_fewer_than_two_values(self):
        assert cs.bootstrap_ci([0.5]) is None
        assert cs.bootstrap_ci([]) is None

    def test_ci_brackets_the_sample_mean_for_constant_values(self):
        # Valori identici -> CI degenere ma deve comunque contenere la media.
        lo, hi = cs.bootstrap_ci([0.5, 0.5, 0.5, 0.5, 0.5], n_resamples=500)
        assert lo == pytest.approx(0.5)
        assert hi == pytest.approx(0.5)

    def test_ci_widens_with_more_variance(self):
        tight = cs.bootstrap_ci([0.500, 0.501, 0.499, 0.500, 0.500], n_resamples=2000)
        wide = cs.bootstrap_ci([0.40, 0.60, 0.45, 0.55, 0.50], n_resamples=2000)
        assert (wide[1] - wide[0]) > (tight[1] - tight[0])


class TestSignificanceTest:
    def test_sign_test_none_when_all_diffs_zero(self):
        assert cs._sign_test([0.0, 0.0, 0.0]) is None

    def test_sign_test_minimum_p_value_at_n5_matches_documented_floor(self):
        # Documentato nel docstring dello script: con n=5 tutte le
        # differenze con lo stesso segno, p minimo = 2*(1/32) = 0.0625.
        diffs = [0.01, 0.02, 0.01, 0.03, 0.015]  # tutte positive
        p = cs._sign_test(diffs)
        assert p == pytest.approx(2 * (1 / 32))

    def test_significance_test_declares_method_matching_current_environment(self):
        # Fix (2026-09-04, trovato da un fallimento reale su una macchina CON
        # scipy — questo sandbox non ce l'ha, la macchina dell'utente sì):
        # la versione precedente di questo test asserava `_SCIPY_AVAILABLE is
        # False` come se fosse un invariante del codice, quando è solo un
        # fatto sull'ambiente in cui gira pytest — falliva su qualunque
        # macchina con scipy installato. significance_test() deve dichiarare
        # SEMPRE il metodo realmente usato (mai in modo silenzioso), qualunque
        # esso sia — questo è l'invariante da testare, non quale ramo è attivo
        # nell'ambiente locale.
        p, method = cs.significance_test([0.51, 0.52, 0.53, 0.54, 0.55])
        assert method == ("wilcoxon" if cs._SCIPY_AVAILABLE else "sign_test")
        assert p is not None

    def test_significance_test_sign_test_path_forced(self, monkeypatch):
        # Copre il ramo sign-test in modo deterministico indipendentemente
        # da quale ambiente esegue il test (scipy presente o assente).
        monkeypatch.setattr(cs, "_SCIPY_AVAILABLE", False)
        p, method = cs.significance_test([0.51, 0.52, 0.53, 0.54, 0.55])
        assert method == "sign_test"
        assert p == pytest.approx(2 * (1 / 32))  # 5/5 sopra 0.5, stesso caso di sopra

    def test_significance_test_wilcoxon_path_skipped_if_scipy_absent(self):
        # Copre il ramo wilcoxon solo se scipy è davvero disponibile — non
        # forzabile con monkeypatch (richiede l'import reale di scipy.stats).
        if not cs._SCIPY_AVAILABLE:
            pytest.skip("scipy non disponibile in questo ambiente")
        p, method = cs.significance_test([0.51, 0.52, 0.53, 0.54, 0.55])
        assert method == "wilcoxon"
        assert p is not None

    def test_significance_test_empty_input(self):
        p, method = cs.significance_test([])
        assert p is None
        assert method == "n/a"


class TestTostEquivalence:
    """TOST (Sprint 10zz+87, 2026-09-14) — su richiesta esplicita di un
    feedback esterno verificato: il framing "non rigettiamo il nulla contro
    AUC=0.5" non e' evidenza di equivalenza (specialmente con n=5, dove il
    sign test non puo' MAI essere significativo — vedi classe sopra). Il
    metodo Schuirmann (CI a due code di livello 1-2*alpha interamente dentro
    il margine) e' verificato qui contro casi costruiti a mano dove il
    risultato e' noto per costruzione."""

    def test_tight_cluster_near_popmean_is_equivalent(self):
        # Valori strettissimi intorno a 0.5, ben dentro il margine di 0.02 —
        # l'IC bootstrap al 90% deve stare comodamente dentro [0.48, 0.52].
        values = [0.499, 0.501, 0.500, 0.4995, 0.5005]
        result = cs.tost_equivalence(values, popmean=0.5, margin=0.02)
        assert result["equivalent"] is True
        assert result["ci"] is not None
        lo, hi = result["ci"]
        assert result["margin_lo"] <= lo <= hi <= result["margin_hi"]

    def test_values_outside_margin_are_not_equivalent(self):
        # Valori chiaramente fuori dal margine dichiarato (central inversion
        # style, es. matched_formula_auc ~0.21) — l'IC non puo' stare dentro
        # [0.48, 0.52].
        values = [0.21, 0.22, 0.20, 0.23, 0.19]
        result = cs.tost_equivalence(values, popmean=0.5, margin=0.02)
        assert result["equivalent"] is False

    def test_wide_spread_straddling_margin_is_not_equivalent(self):
        # IC largo che copre 0.5 ma sborda oltre il margine su almeno un lato
        # — "non rigettiamo AUC=0.5" non implica equivalenza a un margine
        # stretto, esattamente il punto sollevato dal feedback esterno.
        values = [0.40, 0.45, 0.50, 0.55, 0.60]
        result = cs.tost_equivalence(values, popmean=0.5, margin=0.02)
        assert result["equivalent"] is False

    def test_margin_bounds_reported_correctly(self):
        result = cs.tost_equivalence([0.5] * 5, popmean=0.5, margin=0.02)
        assert result["margin_lo"] == pytest.approx(0.48)
        assert result["margin_hi"] == pytest.approx(0.52)

    def test_n_less_than_2_returns_none_ci_not_a_crash(self):
        result = cs.tost_equivalence([0.5], popmean=0.5, margin=0.02)
        assert result["ci"] is None
        assert result["equivalent"] is None

    def test_empty_input_does_not_crash(self):
        result = cs.tost_equivalence([], popmean=0.5, margin=0.02)
        assert result["ci"] is None
        assert result["equivalent"] is None

    def test_custom_margin_and_alpha_reflected_in_output(self):
        result = cs.tost_equivalence([0.5, 0.5, 0.5], popmean=0.5, margin=0.05, alpha=0.1)
        assert result["margin"] == 0.05
        assert result["alpha"] == 0.1
        assert result["margin_lo"] == pytest.approx(0.45)
        assert result["margin_hi"] == pytest.approx(0.55)
