# src/ml/ml_plane.py
# ChargeShield-FL — ML Plane: hub centrale e raccoglitore di artefatti reale
#
# FIX 2026-07-22 (richiesta esplicita dell'autore, dopo la review di coerenza
# col paper QRS 2026 — Imperatrice & Romano, "Toward Realistic Privacy Risk in
# FL-Enabled OT Intrusion Detection Systems"): PRIMA di questo file, il ML
# Plane esisteva SOLO come capacità per-componente (AutoencoderTrainer,
# GradientManager, FedAvgAggregator ereditano `emit_event()`/`subscribe()` da
# AbstractMLModel, vedi base_ml.py) — ciascuno con il proprio registro interno
# di listener, mai collegato a nessun altro componente. `emit_event()` viene
# davvero chiamato durante il training reale (verificato via grep durante la
# review pre-push del 2026-07-22), ma NULLA nella pipeline che produce i
# risultati (scripts/run_experiments.py, chargeshield_aggregator.py per
# NVFLARE) chiamava mai `subscribe()` su nessuno di questi componenti — ogni
# evento veniva emesso nel vuoto (0 listener registrati), un'implementazione
# fedele all'idea del paper QRS ma di fatto morta, mai usata.
#
# Questo modulo introduce due pezzi mancanti:
#
# 1. `MLPlane` — l'hub centrale UNICO descritto in docs/MLPlane.md e nel
#    paper QRS 2026 ("Privacy Auditor module co-located with the aggregation
#    server", Fig. 1/2): un solo oggetto a cui ogni componente emittente si
#    registra (via `wire()`), e a cui si registrano i VERI consumatori (qui,
#    `FLArtifactCollector`) tramite `subscribe()`. Prima di questa classe,
#    ogni componente aveva il proprio registro isolato — un listener doveva
#    iscriversi separatamente a ciascuno (AutoencoderTrainer, GradientManager,
#    FedAvgAggregator) per avere visibilità completa. `MLPlane` implementa
#    esso stesso `MLPlaneListener`, quindi `component.subscribe(mlplane)`
#    funziona senza modificare le classi esistenti.
#
# 2. `FLArtifactCollector` — il consumatore reale che raccoglie gli artefatti
#    (update grezzi pre-DP, update privatizzati/clippati, aggregazioni) nel
#    punto esatto di osservazione descritto dal paper QRS: "client updates
#    are temporarily available in server memory before the execution of
#    FedAvg". Sostituisce, in `run_fl_rounds()`, la costruzione diretta di
#    `fl_results` a partire da variabili Python locali con una raccolta
#    realmente sourced dagli eventi del ML Plane — stessi oggetti (stessa
#    identità, non copie), ma ora davvero "letti" dal ML Plane invece che
#    passati a mano tra funzioni.

from __future__ import annotations

from ml.base_ml import (
    AbstractMLModel,
    AggregatedUpdate,
    GradientUpdate,
    MLPlaneEvent,
    MLPlaneListener,
)


class MLPlane(MLPlaneListener):
    """
    Hub centrale del ML Plane — vedi commento di modulo per il contesto.

    Uso tipico in `run_fl_rounds()`:
        mlplane   = MLPlane()
        collector = FLArtifactCollector()
        mlplane.subscribe(collector)
        mlplane.wire(*trainers.values(), gm, agg)
        # ... da qui in poi, ogni emit_event() di trainer/gm/agg raggiunge
        # collector attraverso mlplane, senza bisogno che nessuno dei tre
        # componenti conosca l'esistenza di collector.
    """

    def __init__(self) -> None:
        self._listeners: list[MLPlaneListener] = []

    def subscribe(self, listener: MLPlaneListener) -> None:
        """Registra un consumatore reale (es. FLArtifactCollector)."""
        self._listeners.append(listener)

    def wire(self, *components: AbstractMLModel) -> None:
        """
        Registra questo MLPlane come listener di ogni componente emittente
        (AutoencoderTrainer, GradientManager, FedAvgAggregator, o qualunque
        AbstractMLModel). Da chiamare una volta per componente, dopo la sua
        creazione — ogni emit_event() successivo del componente raggiungerà
        `on_ml_event()` qui sotto, che ridistribuisce a tutti i subscriber.
        """
        for component in components:
            component.subscribe(self)

    def on_ml_event(self, event: MLPlaneEvent) -> None:
        """Ricevuto da un componente wired — ridistribuito ai subscriber reali."""
        for listener in self._listeners:
            listener.on_ml_event(event)


class FLArtifactCollector(MLPlaneListener):
    """
    Consumatore reale del ML Plane — vedi commento di modulo.

    Raccoglie, round per round e nodo per nodo, tre categorie di artefatti:

    - "raw" (purdue_level=1): l'update grezzo pre-DP emesso da
      `AutoencoderTrainer.train_local()` — il primo punto in cui l'artefatto
      "attraversa il confine" dal training locale del client verso l'esterno,
      nel senso del paper QRS ("model updates... traverse supervisory layers
      and cross the DMZ toward the aggregation server").
    - "privatized" (purdue_level=2): l'update dopo clipping/rumore DP,
      emesso da `GradientManager.privatize()`/`clip_only()` — l'artefatto
      così come verrebbe effettivamente ricevuto da un aggregatore
      honest-but-curious in questa modalità DP.
    - "aggregation": il modello globale aggregato, emesso da
      `FedAvgAggregator.aggregate()`.

    Semantica "ultimo vince" per (round, node_id)/round: se un evento per la
    stessa chiave viene emesso più volte nello stesso round (es. un update
    ri-emesso dopo la scalatura Byzantine in `run_fl_rounds()`, o
    un'aggregazione ri-emessa dopo `privatize_aggregate()` in modalità
    "central" — vedi entrambi i punti in `run_fl_rounds()`), la versione più
    recente sovrascrive la precedente. Questo riflette correttamente "cosa
    arriva davvero al confine di aggregazione", non la prima bozza.
    """

    def __init__(self) -> None:
        self._raw_by_round: dict[int, dict[str, GradientUpdate]] = {}
        self._privatized_by_round: dict[int, dict[str, GradientUpdate]] = {}
        self._aggregation_by_round: dict[int, AggregatedUpdate] = {}

    def on_ml_event(self, event: MLPlaneEvent) -> None:
        if event.event_type == "gradient_upload":
            payload = event.payload
            if not isinstance(payload, GradientUpdate):
                return
            if event.purdue_level == 1:
                self._raw_by_round.setdefault(event.round_num, {})[payload.node_id] = payload
            elif event.purdue_level == 2:
                self._privatized_by_round.setdefault(event.round_num, {})[payload.node_id] = payload
        elif event.event_type == "aggregation":
            if isinstance(event.payload, AggregatedUpdate):
                self._aggregation_by_round[event.round_num] = event.payload

    def raw_updates(self, round_num: int) -> list[GradientUpdate]:
        """Update pre-DP osservati in questo round, ordine di prima emissione per nodo."""
        return list(self._raw_by_round.get(round_num, {}).values())

    def privatized_updates(self, round_num: int) -> list[GradientUpdate]:
        """
        Update post-DP/clip osservati in questo round. Se nessun evento
        purdue_level=2 è stato emesso per questo round (caso `no_dp=True`,
        dove `GradientManager` non viene mai invocato — vedi `run_fl_rounds()`),
        ricade sugli update raw: in quel caso l'update "privatizzato" È
        letteralmente l'update raw invariato, stesso oggetto che
        `run_fl_rounds()` passa oggi a `agg.collect()`.
        """
        privatized = self._privatized_by_round.get(round_num)
        if privatized:
            return list(privatized.values())
        return self.raw_updates(round_num)

    def aggregation(self, round_num: int) -> AggregatedUpdate | None:
        return self._aggregation_by_round.get(round_num)
