# tests/test_privacy_auditor_subscriber.py
"""
Unit Tests — PrivacyAuditorSubscriber (Fase 8, wiring event-driven reale)
==========================================================================
Requisiti: pytest, torch (src/auditor/privacy_auditor_subscriber.py importa
torch a livello di modulo — NON eseguibile in ambienti senza torch, a
differenza di tests/test_ml_plane.py, che copre MLPlane/FLArtifactCollector
senza questo requisito).

Gap segnalato dal deep review round 4 (2026-09-14): nessun test esistente
copriva PrivacyAuditorSubscriber — tests/test_sprint5.py testa i singoli
componenti emittenti, tests/test_privacy_auditor.py testa PrivacyAuditor
in isolamento (chiamato direttamente con un model_update python puro),
ma nessuno dei due testa QUESTA classe, che è il pezzo che li collega
davvero nella pipeline (sim + NVFLARE) — selezione raw/privatized,
normalizzazione peer-relative (mediana → max_grad_norm), avanzamento
baseline SOLO da raw (mai da privatized), idempotenza per round.

Non eseguito in sandbox (torch assente) — da lanciare sulla macchina reale
con: `pytest tests/test_privacy_auditor_subscriber.py -v`
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from auditor.privacy_auditor import PrivacyAuditor
from auditor.privacy_auditor_subscriber import PrivacyAuditorSubscriber
from ml.base_ml import AggregatedUpdate, GradientUpdate, MLPlaneEvent
from ml.ml_plane import FLArtifactCollector


# --- Stub auditor: conta le chiamate e cattura gli argomenti, senza la
# logica reale di PrivacyAuditor — usato per i test che verificano IL
# WIRING (quante volte/con quali argomenti viene chiamato audit()), non il
# comportamento di PrivacyAuditor stesso (già coperto da
# tests/test_privacy_auditor.py). ---

class _StubAuditor:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def audit(self, node_id, round_id, model_update):
        self.calls.append((node_id, round_id, model_update))
        return SimpleNamespace(node_id=node_id, round_id=round_id, model_update=model_update)


def _raw_event(node_id: str, round_num: int, weights: list, n_samples: int = 10) -> MLPlaneEvent:
    update = GradientUpdate(
        node_id=node_id, cluster_id="A", round_num=round_num,
        weights=weights, gradients=None, loss=0.1, n_samples=n_samples,
    )
    return MLPlaneEvent(event_type="gradient_upload", purdue_level=1, payload=update, round_num=round_num)


def _privatized_event(node_id: str, round_num: int, weights: list, n_samples: int = 10) -> MLPlaneEvent:
    update = GradientUpdate(
        node_id=node_id, cluster_id="A", round_num=round_num,
        weights=weights, gradients=None, loss=0.1, n_samples=n_samples,
    )
    return MLPlaneEvent(event_type="gradient_upload", purdue_level=2, payload=update, round_num=round_num)


def _aggregation_event(round_num: int) -> MLPlaneEvent:
    aggregated = AggregatedUpdate(
        round_num=round_num, global_weights=[], n_participants=1, mean_loss=0.0,
    )
    return MLPlaneEvent(event_type="aggregation", purdue_level=3, payload=aggregated, round_num=round_num)


@pytest.fixture
def real_auditor():
    """PrivacyAuditor reale (torch-free al suo interno, ma i model_update che
    riceve da PrivacyAuditorSubscriber contengono torch.Tensor — vedi
    _flatten_model_update in privacy_auditor.py, che li gestisce via duck
    typing hasattr('flatten')/hasattr('tolist'), nessun import torch lì)."""
    auditor = PrivacyAuditor(config_path="config/auditor.yaml")
    auditor.reset()
    return auditor


# --- on_ml_event: filtro sul tipo di evento ---

class TestOnMlEventFiltering:

    def test_ignores_non_aggregation_events(self, real_auditor):
        """Un evento 'gradient_upload' NON deve triggerare _handle_round_complete
        — solo 'aggregation' lo fa (il segnale 'round completo')."""
        collector = FLArtifactCollector()
        subscriber = PrivacyAuditorSubscriber(real_auditor, collector, max_grad_norm=1.0)

        subscriber.on_ml_event(_raw_event("n1", round_num=1, weights=[torch.tensor([1.0, 2.0])]))

        assert subscriber.reports_for_round(1) == {}
        assert subscriber.gradients_for_round(1) == {}

    def test_aggregation_event_triggers_processing(self, real_auditor):
        collector = FLArtifactCollector()
        collector.on_ml_event(_raw_event("n1", round_num=1, weights=[torch.tensor([1.0, 2.0])]))
        subscriber = PrivacyAuditorSubscriber(real_auditor, collector, max_grad_norm=1.0)

        subscriber.on_ml_event(_aggregation_event(round_num=1))

        assert "n1" in subscriber.reports_for_round(1)


# --- Flusso base: raw updates -> report per nodo ---

class TestBasicFlow:

    def test_produces_report_and_gradients_per_node(self, real_auditor):
        collector = FLArtifactCollector()
        collector.on_ml_event(_raw_event("n1", round_num=1, weights=[torch.tensor([1.0, 2.0])]))
        collector.on_ml_event(_raw_event("n2", round_num=1, weights=[torch.tensor([3.0, 4.0])]))
        subscriber = PrivacyAuditorSubscriber(real_auditor, collector, max_grad_norm=1.0)

        subscriber.on_ml_event(_aggregation_event(round_num=1))

        reports = subscriber.reports_for_round(1)
        gradients = subscriber.gradients_for_round(1)
        assert set(reports.keys()) == {"n1", "n2"}
        assert set(gradients.keys()) == {"n1", "n2"}
        assert "layer_0" in gradients["n1"]

    def test_no_updates_for_round_produces_empty_reports(self, real_auditor):
        """Un round senza nessun update raccolto (né raw né privatized) deve dare
        reports/gradients vuoti, senza sollevare eccezioni."""
        collector = FLArtifactCollector()
        subscriber = PrivacyAuditorSubscriber(real_auditor, collector, max_grad_norm=1.0)

        subscriber.on_ml_event(_aggregation_event(round_num=5))

        assert subscriber.reports_for_round(5) == {}
        assert subscriber.gradients_for_round(5) == {}

    def test_missing_round_getters_return_empty_dict_not_none(self, real_auditor):
        collector = FLArtifactCollector()
        subscriber = PrivacyAuditorSubscriber(real_auditor, collector, max_grad_norm=1.0)
        assert subscriber.reports_for_round(999) == {}
        assert subscriber.gradients_for_round(999) == {}

    def test_update_with_falsy_node_id_is_skipped(self, real_auditor):
        """Un update con node_id vuoto ('') deve essere escluso dal processing —
        non deve comparire nei report né causare un KeyError altrove."""
        collector = FLArtifactCollector()
        collector.on_ml_event(_raw_event("", round_num=1, weights=[torch.tensor([1.0])]))
        collector.on_ml_event(_raw_event("n1", round_num=1, weights=[torch.tensor([2.0])]))
        subscriber = PrivacyAuditorSubscriber(real_auditor, collector, max_grad_norm=1.0)

        subscriber.on_ml_event(_aggregation_event(round_num=1))

        assert set(subscriber.reports_for_round(1).keys()) == {"n1"}


# --- Idempotenza per round ---

class TestIdempotency:

    def test_same_round_processed_only_once(self):
        """Un'aggregazione ri-emessa per lo STESSO round_num (es. central DP,
        dopo privatize_aggregate()) deve essere un no-op esplicito la seconda
        volta — audit() non deve essere richiamato di nuovo."""
        stub = _StubAuditor()
        collector = FLArtifactCollector()
        collector.on_ml_event(_raw_event("n1", round_num=1, weights=[torch.tensor([1.0, 2.0])]))
        subscriber = PrivacyAuditorSubscriber(stub, collector, max_grad_norm=1.0)

        subscriber.on_ml_event(_aggregation_event(round_num=1))
        subscriber.on_ml_event(_aggregation_event(round_num=1))  # ri-emesso, stesso round

        assert len(stub.calls) == 1, (
            f"audit() atteso 1 volta, chiamato {len(stub.calls)} — l'idempotenza "
            "per round sembra rotta"
        )

    def test_different_rounds_both_processed(self):
        stub = _StubAuditor()
        collector = FLArtifactCollector()
        collector.on_ml_event(_raw_event("n1", round_num=1, weights=[torch.tensor([1.0])]))
        collector.on_ml_event(_raw_event("n1", round_num=2, weights=[torch.tensor([2.0])]))
        subscriber = PrivacyAuditorSubscriber(stub, collector, max_grad_norm=1.0)

        subscriber.on_ml_event(_aggregation_event(round_num=1))
        subscriber.on_ml_event(_aggregation_event(round_num=2))

        assert len(stub.calls) == 2


# --- Selezione raw vs privatized ---

class TestRawVsPrivatizedSelection:

    def test_uses_privatized_when_raw_empty(self):
        """Se non arriva mai un evento purdue_level=1 per il round (no_dp=False,
        GradientManager sempre invocato), il subscriber deve processare comunque
        usando gli update privatizzati — non deve restare vuoto."""
        stub = _StubAuditor()
        collector = FLArtifactCollector()
        collector.on_ml_event(_privatized_event("n1", round_num=1, weights=[torch.tensor([9.0])]))
        subscriber = PrivacyAuditorSubscriber(stub, collector, max_grad_norm=1.0)

        subscriber.on_ml_event(_aggregation_event(round_num=1))

        assert len(stub.calls) == 1
        assert "n1" in subscriber.reports_for_round(1)

    def test_prefers_raw_over_privatized_when_both_present(self):
        """Se ESISTONO entrambi raw e privatized per lo stesso round, il delta deve
        derivare dal raw (vista più informativa), non dal privatizzato — verificato
        tramite il valore del gradiente catturato (non scalato) per il nodo."""
        stub = _StubAuditor()
        collector = FLArtifactCollector()
        collector.on_ml_event(_raw_event("n1", round_num=1, weights=[torch.tensor([10.0])]))
        collector.on_ml_event(_privatized_event("n1", round_num=1, weights=[torch.tensor([-999.0])]))
        subscriber = PrivacyAuditorSubscriber(stub, collector, max_grad_norm=1.0)

        subscriber.on_ml_event(_aggregation_event(round_num=1))

        grad = subscriber.gradients_for_round(1)["n1"]["layer_0"]
        assert float(grad) == pytest.approx(10.0), (
            "il delta dovrebbe derivare dal raw (10.0), non dal privatized (-999.0) "
            f"— trovato {float(grad)}"
        )


# --- Baseline: avanza SOLO da raw, mai da privatized ---

class TestBaselineAdvancement:

    def test_baseline_advances_after_round_with_raw_updates(self):
        """Dopo un round con update RAW, la baseline deve avanzare alla loro media
        pesata — verificato osservando il delta del round successivo."""
        stub = _StubAuditor()
        collector = FLArtifactCollector()
        subscriber = PrivacyAuditorSubscriber(stub, collector, max_grad_norm=1.0)
        subscriber.set_initial_baseline([torch.tensor([0.0])])

        collector.on_ml_event(_raw_event("n1", round_num=1, weights=[torch.tensor([4.0])]))
        subscriber.on_ml_event(_aggregation_event(round_num=1))

        collector.on_ml_event(_raw_event("n1", round_num=2, weights=[torch.tensor([9.0])]))
        subscriber.on_ml_event(_aggregation_event(round_num=2))

        grad_r2 = subscriber.gradients_for_round(2)["n1"]["layer_0"]
        assert float(grad_r2) == pytest.approx(5.0), (
            "delta round 2 atteso 9.0 - 4.0 = 5.0 (baseline avanzata alla media "
            f"raw del round 1) — trovato {float(grad_r2)}"
        )

    def test_baseline_does_not_advance_from_privatized_only_round(self):
        """Se un round ha SOLO update privatized (nessun raw — es. dp_mode='local',
        dove GradientManager.accept() non emette mai purdue_level=1, vedi commento
        in privacy_auditor_subscriber.py), la baseline non deve avanzare — il round
        successivo deve calcolare il delta contro la baseline ORIGINALE, non contro
        i dati (privatizzati) del round intermedio. Regressione diretta sul limite
        noto 'IDS usa pesi PRE-DP... sotto local DP la baseline non avanza'."""
        stub = _StubAuditor()
        collector = FLArtifactCollector()
        subscriber = PrivacyAuditorSubscriber(stub, collector, max_grad_norm=1.0)
        subscriber.set_initial_baseline([torch.tensor([0.0, 0.0])])

        # Round 1: SOLO privatized (simula dp_mode="local")
        collector.on_ml_event(_privatized_event("n1", round_num=1, weights=[torch.tensor([5.0, 5.0])]))
        subscriber.on_ml_event(_aggregation_event(round_num=1))

        # Round 2: arriva un raw update — il delta deve essere calcolato contro la
        # baseline ORIGINALE [0,0], non contro [5,5] del round 1.
        collector.on_ml_event(_raw_event("n1", round_num=2, weights=[torch.tensor([8.0, 8.0])]))
        subscriber.on_ml_event(_aggregation_event(round_num=2))

        grad_r2 = subscriber.gradients_for_round(2)["n1"]["layer_0"]
        assert torch.allclose(grad_r2, torch.tensor([8.0, 8.0])), (
            "la baseline sembra essere avanzata erroneamente da un round "
            f"solo-privatized — delta atteso [8.0, 8.0], trovato {grad_r2.tolist()}"
        )

    def test_no_baseline_set_uses_raw_weights_directly_as_delta(self):
        """Senza set_initial_baseline() (default None), il primo round deve usare
        i pesi grezzi COME delta (nessuna sottrazione) — comportamento esplicito
        per il round 1 in NVFLARE (KNOWN GAP documentato: baseline resta None)."""
        stub = _StubAuditor()
        collector = FLArtifactCollector()
        collector.on_ml_event(_raw_event("n1", round_num=1, weights=[torch.tensor([3.0, 1.0])]))
        subscriber = PrivacyAuditorSubscriber(stub, collector, max_grad_norm=1.0)
        # nessun set_initial_baseline() chiamato — _prev_baseline resta None

        subscriber.on_ml_event(_aggregation_event(round_num=1))

        grad = subscriber.gradients_for_round(1)["n1"]["layer_0"]
        assert torch.allclose(grad, torch.tensor([3.0, 1.0]))


# --- Normalizzazione peer-relative (scala mediana -> max_grad_norm) ---

class TestMedianScale:

    def test_scale_applied_using_median_norm_across_clients(self):
        """Con 3 client di norma nota (1, 5, 10), la scala deve usare la MEDIANA
        (5), non media né max/min — model_update passato all'auditor per il
        client di norma 5 deve essere moltiplicato per max_grad_norm/5."""
        stub = _StubAuditor()
        collector = FLArtifactCollector()
        # norme: n1=5 (3-4-5), n2=10 (6-8-10), n3=1 (0.6-0.8-1) — mediana=5
        collector.on_ml_event(_raw_event("n1", round_num=1, weights=[torch.tensor([3.0, 4.0])]))
        collector.on_ml_event(_raw_event("n2", round_num=1, weights=[torch.tensor([6.0, 8.0])]))
        collector.on_ml_event(_raw_event("n3", round_num=1, weights=[torch.tensor([0.6, 0.8])]))
        subscriber = PrivacyAuditorSubscriber(stub, collector, max_grad_norm=2.0)

        subscriber.on_ml_event(_aggregation_event(round_num=1))

        # scale atteso = max_grad_norm / mediana_norme = 2.0 / 5.0 = 0.4
        call_by_node = {c[0]: c for c in stub.calls}
        model_update_n1 = call_by_node["n1"][2]
        scaled = model_update_n1["layer_0"]
        assert torch.allclose(scaled, torch.tensor([3.0, 4.0]) * 0.4, atol=1e-5), (
            f"model_update scalato atteso [1.2, 1.6], trovato {scaled.tolist()}"
        )

    def test_scale_defaults_to_one_when_median_norm_near_zero(self):
        """Se la norma mediana è sotto la soglia 1e-4 (client con pesi ~zero), la
        scala deve ricadere su 1.0 esplicito, MAI dividere per un numero
        vicino a zero (eviterebbe un'esplosione numerica del model_update)."""
        stub = _StubAuditor()
        collector = FLArtifactCollector()
        collector.on_ml_event(_raw_event("n1", round_num=1, weights=[torch.tensor([0.0, 0.0])]))
        subscriber = PrivacyAuditorSubscriber(stub, collector, max_grad_norm=2.0)

        subscriber.on_ml_event(_aggregation_event(round_num=1))

        model_update_n1 = stub.calls[0][2]
        scaled = model_update_n1["layer_0"]
        assert torch.allclose(scaled, torch.tensor([0.0, 0.0])), (
            "con scale=1.0 (fallback) il model_update deve restare invariato "
            f"(zero), trovato {scaled.tolist()}"
        )


# --- _weighted_average (metodo statico, stessa formula dell'aggregator) ---

class TestWeightedAverage:

    def test_weighted_average_matches_n_samples_proportion(self):
        u1 = GradientUpdate(node_id="n1", cluster_id="A", round_num=1,
                             weights=[torch.tensor([10.0])], gradients=None, loss=0.1, n_samples=100)
        u2 = GradientUpdate(node_id="n2", cluster_id="A", round_num=1,
                             weights=[torch.tensor([0.0])], gradients=None, loss=0.1, n_samples=300)
        averaged = PrivacyAuditorSubscriber._weighted_average([u1, u2])
        # media pesata: 10.0*(100/400) + 0.0*(300/400) = 2.5
        # (weights[i] è un tensore 1-elemento -> averaged[0] ha shape (1,))
        assert torch.allclose(averaged[0], torch.tensor([2.5]))

    def test_weighted_average_empty_list_returns_none(self):
        assert PrivacyAuditorSubscriber._weighted_average([]) is None
