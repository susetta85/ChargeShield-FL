# tests/test_ml_plane.py
"""
Unit Tests — MLPlane (hub centrale) e FLArtifactCollector
==========================================================
Gap segnalato dal deep review round 4 (2026-09-14): src/ml/ml_plane.py
non aveva NESSUN test dedicato — tests/test_sprint5.py verifica solo che i
singoli componenti (AutoencoderTrainer/GradientManager/FedAvgAggregator)
emettano eventi correttamente sottoscrivendo uno Spy direttamente a LORO,
mai la propagazione tramite l'hub MLPlane né la logica di raccolta di
FLArtifactCollector (raw preferito, fallback privatizzato, "ultimo vince"
per round/nodo) — cioè esattamente il pezzo di wiring event-driven che il
Privacy Auditor (src/auditor/privacy_auditor_subscriber.py) e IDS
consumano a valle.

Questo file NON richiede torch: src/ml/ml_plane.py è puro Python (nessun
`import torch` a livello di modulo — la classe manipola solo dict/list di
GradientUpdate/AggregatedUpdate, mai i tensori al loro interno), quindi
questi test possono girare anche in ambienti senza torch installato.
I test di PrivacyAuditorSubscriber (che DEVE importare torch per le
operazioni sui pesi) sono in un file separato,
tests/test_privacy_auditor_subscriber.py, eseguibile solo dove torch è
disponibile.

Cosa testiamo:
- MLPlane.wire()/subscribe(): un evento emesso da un componente wired
  raggiunge ogni subscriber registrato.
- FLArtifactCollector: raccolta separata raw (purdue_level=1) vs
  privatized (purdue_level=2) vs aggregation; fallback privatized→raw
  quando nessun evento purdue_level=2 è mai stato emesso per il round;
  semantica "ultimo vince" per lo stesso (round, node_id).
- Eventi di tipo/round/node diversi non si mescolano tra loro.

Cosa NON testiamo qui:
- PrivacyAuditorSubscriber (richiede torch, vedi file separato).
- L'uso reale in run_fl_rounds()/chargeshield_aggregator.py (già coperto
  da test_run_experiments_integration.py/test_flare_connector.py).
"""

import pytest

from ml.base_ml import (
    AbstractMLModel,
    AggregatedUpdate,
    GradientUpdate,
    MLPlaneEvent,
    MLPlaneListener,
)
from ml.ml_plane import FLArtifactCollector, MLPlane


# --- Fixture: componente emittente minimale (implementa AbstractMLModel) ---

class _FakeComponent(AbstractMLModel):
    """Componente ML Plane minimale, sufficiente per emettere eventi verso
    i listener registrati — non fa training reale, solo wiring."""

    def __init__(self) -> None:
        self._listeners: list[MLPlaneListener] = []

    def get_weights(self):
        return []

    def set_weights(self, weights) -> None:
        pass

    def train_step(self, data) -> float:
        return 0.0

    def emit_event(self, event: MLPlaneEvent) -> None:
        for listener in self._listeners:
            listener.on_ml_event(event)

    def subscribe(self, listener: MLPlaneListener) -> None:
        self._listeners.append(listener)


class _SpyListener(MLPlaneListener):
    def __init__(self) -> None:
        self.events: list[MLPlaneEvent] = []

    def on_ml_event(self, event: MLPlaneEvent) -> None:
        self.events.append(event)


def _make_update(node_id: str, round_num: int, weights=None, n_samples: int = 10) -> GradientUpdate:
    return GradientUpdate(
        node_id=node_id,
        cluster_id="A",
        round_num=round_num,
        weights=weights if weights is not None else [1.0, 2.0],
        gradients=None,
        loss=0.1,
        n_samples=n_samples,
    )


def _make_aggregated(round_num: int, weights=None) -> AggregatedUpdate:
    return AggregatedUpdate(
        round_num=round_num,
        global_weights=weights if weights is not None else [1.5],
        n_participants=2,
        mean_loss=0.05,
    )


# --- MLPlane: wiring hub-and-subscriber ---

class TestMLPlaneWiring:

    def test_wire_forwards_events_to_subscriber(self):
        """Un evento emesso da un componente wired deve raggiungere il subscriber
        registrato su MLPlane, senza che il componente conosca il subscriber."""
        mlplane = MLPlane()
        spy = _SpyListener()
        component = _FakeComponent()

        mlplane.subscribe(spy)
        mlplane.wire(component)

        update = _make_update("n1", round_num=1)
        component.emit_event(MLPlaneEvent(
            event_type="gradient_upload", purdue_level=1, payload=update, round_num=1,
        ))

        assert len(spy.events) == 1
        assert spy.events[0].payload is update

    def test_wire_supports_multiple_components(self):
        """wire() deve accettare più componenti in una sola chiamata — ogni
        emit_event() di ciascuno deve raggiungere lo stesso subscriber."""
        mlplane = MLPlane()
        spy = _SpyListener()
        c1, c2 = _FakeComponent(), _FakeComponent()

        mlplane.subscribe(spy)
        mlplane.wire(c1, c2)

        c1.emit_event(MLPlaneEvent(event_type="gradient_upload", purdue_level=1,
                                    payload=_make_update("n1", 1), round_num=1))
        c2.emit_event(MLPlaneEvent(event_type="aggregation", purdue_level=3,
                                    payload=_make_aggregated(1), round_num=1))

        assert len(spy.events) == 2
        assert {e.event_type for e in spy.events} == {"gradient_upload", "aggregation"}

    def test_subscribe_supports_multiple_listeners(self):
        """Più subscriber devono ricevere TUTTI lo stesso evento (broadcast, non round-robin)."""
        mlplane = MLPlane()
        spy1, spy2 = _SpyListener(), _SpyListener()
        component = _FakeComponent()

        mlplane.subscribe(spy1)
        mlplane.subscribe(spy2)
        mlplane.wire(component)

        component.emit_event(MLPlaneEvent(event_type="gradient_upload", purdue_level=1,
                                           payload=_make_update("n1", 1), round_num=1))

        assert len(spy1.events) == 1
        assert len(spy2.events) == 1

    def test_no_subscribers_does_not_raise(self):
        """wire() senza nessun subscribe() precedente non deve sollevare eccezioni
        (nessun listener registrato — l'evento viene semplicemente perso, non è un errore)."""
        mlplane = MLPlane()
        component = _FakeComponent()
        mlplane.wire(component)
        component.emit_event(MLPlaneEvent(event_type="gradient_upload", purdue_level=1,
                                           payload=_make_update("n1", 1), round_num=1))
        # nessuna eccezione = successo


# --- FLArtifactCollector: raccolta raw/privatized/aggregation ---

class TestFLArtifactCollector:

    def test_raw_update_collected_at_purdue_level_1(self):
        collector = FLArtifactCollector()
        update = _make_update("n1", round_num=1)
        collector.on_ml_event(MLPlaneEvent(
            event_type="gradient_upload", purdue_level=1, payload=update, round_num=1,
        ))
        assert collector.raw_updates(1) == [update]

    def test_privatized_update_collected_at_purdue_level_2(self):
        collector = FLArtifactCollector()
        update = _make_update("n1", round_num=1, weights=[0.5])
        collector.on_ml_event(MLPlaneEvent(
            event_type="gradient_upload", purdue_level=2, payload=update, round_num=1,
        ))
        assert collector.privatized_updates(1) == [update]
        # raw resta vuoto: nessun evento purdue_level=1 è mai stato emesso
        assert collector.raw_updates(1) == []

    def test_privatized_falls_back_to_raw_when_no_dp(self):
        """Se non arriva mai un evento purdue_level=2 per il round (caso no_dp=True,
        GradientManager mai invocato), privatized_updates() deve ricadere sul raw —
        stesso oggetto, non una copia."""
        collector = FLArtifactCollector()
        raw = _make_update("n1", round_num=1)
        collector.on_ml_event(MLPlaneEvent(
            event_type="gradient_upload", purdue_level=1, payload=raw, round_num=1,
        ))
        assert collector.privatized_updates(1) == [raw]
        assert collector.privatized_updates(1)[0] is raw

    def test_privatized_does_not_fall_back_when_privatized_present(self):
        """Se ESISTE un evento purdue_level=2, privatized_updates() deve restituire
        quello, non il fallback raw — anche se entrambi sono presenti per lo stesso round."""
        collector = FLArtifactCollector()
        raw = _make_update("n1", round_num=1, weights=[1.0])
        privatized = _make_update("n1", round_num=1, weights=[0.3])
        collector.on_ml_event(MLPlaneEvent(event_type="gradient_upload", purdue_level=1,
                                            payload=raw, round_num=1))
        collector.on_ml_event(MLPlaneEvent(event_type="gradient_upload", purdue_level=2,
                                            payload=privatized, round_num=1))
        result = collector.privatized_updates(1)
        assert result == [privatized]
        assert result[0].weights == [0.3]

    def test_aggregation_collected(self):
        collector = FLArtifactCollector()
        aggregated = _make_aggregated(round_num=1)
        collector.on_ml_event(MLPlaneEvent(
            event_type="aggregation", purdue_level=3, payload=aggregated, round_num=1,
        ))
        assert collector.aggregation(1) is aggregated

    def test_aggregation_missing_round_returns_none(self):
        collector = FLArtifactCollector()
        assert collector.aggregation(99) is None

    def test_last_wins_same_round_and_node(self):
        """Due update raw per lo stesso (round, node_id) — es. un update ri-emesso
        dopo la scalatura Byzantine — il più recente deve sovrascrivere il precedente,
        non accumularsi in una lista."""
        collector = FLArtifactCollector()
        first = _make_update("n1", round_num=1, weights=[1.0])
        second = _make_update("n1", round_num=1, weights=[2.0])
        collector.on_ml_event(MLPlaneEvent(event_type="gradient_upload", purdue_level=1,
                                            payload=first, round_num=1))
        collector.on_ml_event(MLPlaneEvent(event_type="gradient_upload", purdue_level=1,
                                            payload=second, round_num=1))
        result = collector.raw_updates(1)
        assert len(result) == 1
        assert result[0].weights == [2.0]

    def test_different_nodes_same_round_both_kept(self):
        """Due nodi diversi nello stesso round devono coesistere (non è una chiave
        solo-round, ma (round, node_id))."""
        collector = FLArtifactCollector()
        u1 = _make_update("n1", round_num=1)
        u2 = _make_update("n2", round_num=1)
        collector.on_ml_event(MLPlaneEvent(event_type="gradient_upload", purdue_level=1,
                                            payload=u1, round_num=1))
        collector.on_ml_event(MLPlaneEvent(event_type="gradient_upload", purdue_level=1,
                                            payload=u2, round_num=1))
        result = collector.raw_updates(1)
        assert len(result) == 2
        assert {u.node_id for u in result} == {"n1", "n2"}

    def test_rounds_do_not_mix(self):
        """Gli update di round diversi non devono comparire nella query dell'altro round."""
        collector = FLArtifactCollector()
        u_round1 = _make_update("n1", round_num=1)
        u_round2 = _make_update("n1", round_num=2)
        collector.on_ml_event(MLPlaneEvent(event_type="gradient_upload", purdue_level=1,
                                            payload=u_round1, round_num=1))
        collector.on_ml_event(MLPlaneEvent(event_type="gradient_upload", purdue_level=1,
                                            payload=u_round2, round_num=2))
        assert collector.raw_updates(1) == [u_round1]
        assert collector.raw_updates(2) == [u_round2]

    def test_missing_round_returns_empty_list_not_none(self):
        """Query su un round mai visto deve restituire lista vuota, mai None
        (chi consuma raw_updates()/privatized_updates() itera direttamente il risultato)."""
        collector = FLArtifactCollector()
        assert collector.raw_updates(42) == []
        assert collector.privatized_updates(42) == []

    def test_ignores_gradient_upload_with_non_gradientupdate_payload(self):
        """Un evento 'gradient_upload' con payload che non è un GradientUpdate
        (es. None, o un tipo inatteso) deve essere ignorato silenziosamente,
        non sollevare un'eccezione né essere raccolto."""
        collector = FLArtifactCollector()
        collector.on_ml_event(MLPlaneEvent(
            event_type="gradient_upload", purdue_level=1, payload=None, round_num=1,
        ))
        assert collector.raw_updates(1) == []

    def test_ignores_aggregation_with_non_aggregatedupdate_payload(self):
        collector = FLArtifactCollector()
        collector.on_ml_event(MLPlaneEvent(
            event_type="aggregation", purdue_level=3, payload=None, round_num=1,
        ))
        assert collector.aggregation(1) is None

    def test_ignores_unknown_event_type(self):
        """Un event_type non riconosciuto (né 'gradient_upload' né 'aggregation',
        es. 'weight_download') non deve finire in nessuna delle tre categorie raccolte."""
        collector = FLArtifactCollector()
        collector.on_ml_event(MLPlaneEvent(
            event_type="weight_download", purdue_level=3,
            payload=_make_aggregated(1), round_num=1,
        ))
        assert collector.raw_updates(1) == []
        assert collector.privatized_updates(1) == []
        assert collector.aggregation(1) is None


# --- Integrazione: MLPlane + FLArtifactCollector wired insieme ---

class TestMLPlaneWithCollector:

    def test_end_to_end_wiring_matches_direct_collector_use(self):
        """Passare per MLPlane (component -> mlplane -> collector) deve dare
        esattamente lo stesso risultato di chiamare collector.on_ml_event()
        direttamente — MLPlane è un puro inoltro, non deve alterare né
        duplicare l'evento."""
        mlplane = MLPlane()
        collector = FLArtifactCollector()
        component = _FakeComponent()

        mlplane.subscribe(collector)
        mlplane.wire(component)

        update = _make_update("n1", round_num=1, weights=[7.0])
        component.emit_event(MLPlaneEvent(
            event_type="gradient_upload", purdue_level=1, payload=update, round_num=1,
        ))

        assert collector.raw_updates(1) == [update]

    def test_multiple_components_feed_same_collector(self):
        """Più componenti (es. trainer di client diversi) wired sullo stesso
        MLPlane devono finire tutti nello stesso FLArtifactCollector — è
        esattamente il pattern usato in run_fl_rounds()."""
        mlplane = MLPlane()
        collector = FLArtifactCollector()
        trainer_a, trainer_b, aggregator = _FakeComponent(), _FakeComponent(), _FakeComponent()

        mlplane.subscribe(collector)
        mlplane.wire(trainer_a, trainer_b, aggregator)

        trainer_a.emit_event(MLPlaneEvent(event_type="gradient_upload", purdue_level=1,
                                           payload=_make_update("a", 1), round_num=1))
        trainer_b.emit_event(MLPlaneEvent(event_type="gradient_upload", purdue_level=1,
                                           payload=_make_update("b", 1), round_num=1))
        aggregator.emit_event(MLPlaneEvent(event_type="aggregation", purdue_level=3,
                                            payload=_make_aggregated(1), round_num=1))

        assert {u.node_id for u in collector.raw_updates(1)} == {"a", "b"}
        assert collector.aggregation(1) is not None
