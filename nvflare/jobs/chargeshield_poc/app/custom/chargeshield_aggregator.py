# nvflare/jobs/chargeshield_poc/app/custom/chargeshield_aggregator.py
"""
ChargeShieldAggregator — NVFLARE server-side Aggregator custom (fase 2, 2026-07-22).

STATO: scritto e ragionato manualmente in un sandbox dove `nvflare`/`torch`
non sono installabili (stesso limite di chargeshield_executor.py — vedi
docs/NVFlareIntegration.md). NON eseguito, NON testato.

Cosa fa (fase 2 — sostituisce l'aggregatore built-in con le classi VERE e
già testate della simulazione, invece di InTimeAccumulateWeightedAggregator):
    1. accept(): riceve lo Shareable/DXO di ogni client per il round corrente,
       lo converte in GradientUpdate e lo accumula (NESSUN DP qui — fase 3,
       vedi sotto).
    2. aggregate(): quando tutti i client hanno risposto,
       a) chiama FedAvgAggregator.collect()/.aggregate() — la STESSA classe
          usata da scripts/run_experiments.py::run_fl_rounds() in simulazione,
          non una re-implementazione (src/ml/fedavg_aggregator.py).
       b) esegue l'equivalente di run_ids() — PrivacyAuditor.audit() per ogni
          client + ChargingIDS.analyze_round() sul round completo — con la
          STESSA logica di normalizzazione peer-relative (mediana) e lo
          stesso calcolo del delta rispetto al modello del round precedente
          usati in run_experiments.py::run_ids(). Vedi quella funzione per la
          spiegazione completa (fix Sprint 9: GRADIENT_EXPLOSION, Krum).
       c) converte l'AggregatedUpdate risultante in un DXO/Shareable per
          FullModelShareableGenerator.

Cosa NON fa ancora (fase 3, vedi docs/NVFlareIntegration.md):
    - Nessun DP (GradientManager.privatize()/clip_only()/privatize_aggregate()
      non sono chiamati né qui né in chargeshield_executor.py). Gli update
      raccolti in accept() sono quindi "raw" per definizione in questa fase —
      non serve distinguere raw_updates/updates come nella simulazione finché
      la DP non è cablata.
    - I risultati di PrivacyAuditor/ChargingIDS sono solo loggati (logger.info/
      warning), non esportati in un formato strutturato (JSON/Excel) come fa
      save_results() nella simulazione — da aggiungere quando questo gira
      davvero (probabile: scrivere su file via fl_ctx o un componente NVFLARE
      dedicato, non deciso).
    - Non gestisce round con partecipazione parziale (drop-out) — stesso
      limite già segnalato nella review scalabilità/realismo di oggi per
      FedAvgAggregator (min_participants=tutti i client).

Punti VERIFY: (stessa cautela di chargeshield_executor.py — marcati inline):
    - Formato di dxo.data in ingresso (assunto identico a quanto emesso da
      chargeshield_executor.py: dict[str state_dict_key, np.ndarray]).
    - Come ottenere il node/cluster id di un client in accept() — qui uso
      dxo.meta["cluster_id"] (che chargeshield_executor.py imposta), NON
      fl_ctx/peer_ctx — da confermare che sia accessibile in questo punto
      del ciclo di vita di un Aggregator NVFLARE.
    - round_num: nessun contatore affidabile da fl_ctx trovato in questa fase
      di scrittura (stesso limite dell'executor) — uso un contatore locale
      incrementato ad ogni aggregate(), quasi certamente da sostituire con
      AppConstants.CURRENT_ROUND letto da fl_ctx.
    - Il formato di ritorno atteso da aggregate() — qui restituisco un DXO
      DataKind.WEIGHTS con i pesi aggregati completi, assumendo che
      FullModelShareableGenerator lo interpreti come il nuovo modello globale
      (coerente con come InTimeAccumulateWeightedAggregator viene usato nella
      config originale) — da verificare contro la vera classe base
      `nvflare.app_common.abstract.aggregator.Aggregator` una volta
      installabile nvflare.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # .../ChargeShield-FL
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nvflare.apis.dxo import DXO, DataKind, from_shareable  # noqa: E402
from nvflare.apis.fl_context import FLContext  # noqa: E402
from nvflare.apis.shareable import Shareable  # noqa: E402
from nvflare.app_common.abstract.aggregator import Aggregator  # noqa: E402

logger = logging.getLogger(__name__)


class ChargeShieldAggregator(Aggregator):
    """
    Sostituisce InTimeAccumulateWeightedAggregator con FedAvgAggregator +
    PrivacyAuditor + ChargingIDS reali (fase 2, vedi docstring del modulo).

    Args (da config_fed_server.json):
        auditor_config_path: path a config/auditor.yaml (stesso file YAML
                     usato dalla simulazione — nessun parametro hardcoded qui).
        min_clients: numero minimo di client per procedere con l'aggregazione
                     (passato a FedAvgAggregator come min_participants).
        max_grad_norm: necessario per la normalizzazione peer-relative
                     dell'IDS (stessa formula di run_ids()) — deve combaciare
                     con max_grad_norm usato per la DP quando verrà cablata
                     (fase 3), altrimenti la soglia GRADIENT_EXPLOSION non è
                     calibrata correttamente.
        epsilon / explosion_threshold: passati a PrivacyAuditor — vedi
                     run_ids() per la semantica (no_dp non esiste ancora qui,
                     fase 2 è sempre "senza DP" per costruzione).
        byzantine_tolerance / cosine_threshold / krum_threshold: passati a
                     ChargingIDS — stessi default di run_ids() (0, 0.3, 3.5).
    """

    def __init__(
        self,
        auditor_config_path: str = "config/auditor.yaml",
        min_clients: int = 4,
        max_grad_norm: float = 1.0,
        epsilon: float | None = None,
        explosion_threshold: float | None = None,
        byzantine_tolerance: int = 0,
        cosine_threshold: float = 0.3,
        krum_threshold: float = 3.5,
    ):
        super().__init__()
        self._auditor_config_path = str(_PROJECT_ROOT / auditor_config_path)
        self._min_clients = min_clients
        self._max_grad_norm = max_grad_norm
        self._epsilon = epsilon
        self._explosion_threshold = explosion_threshold
        self._byzantine_tolerance = byzantine_tolerance
        self._cosine_threshold = cosine_threshold
        self._krum_threshold = krum_threshold

        # Istanziati lazy in _ensure_components() — evita di importare
        # torch/ml/auditor/ids al momento della definizione della classe
        # (che NVFLARE può introspezionare prima di START_RUN).
        self._fedavg = None
        self._auditor = None
        self._ids = None

        self._weight_keys: list[str] | None = None
        self._round_updates: list[Any] = []   # GradientUpdate raccolti questo round
        self._round_num = 0                    # VERIFY: leggere da fl_ctx, non contare localmente
        self._prev_raw_global: list[Any] | None = None  # per il delta peer-relative IDS

    # ── Lazy init ────────────────────────────────────────────────────────────

    def _ensure_components(self) -> None:
        if self._fedavg is not None:
            return
        from ml.fedavg_aggregator import FedAvgAggregator
        from auditor.privacy_auditor import PrivacyAuditor
        from ids.charging_ids import ChargingIDS

        self._fedavg = FedAvgAggregator({"min_participants": self._min_clients})
        self._auditor = PrivacyAuditor(
            config_path=self._auditor_config_path,
            epsilon=self._epsilon,
            explosion_threshold=self._explosion_threshold,
        )
        self._ids = ChargingIDS(
            config_path=self._auditor_config_path,
            byzantine_tolerance=self._byzantine_tolerance,
            cosine_threshold=self._cosine_threshold,
            krum_threshold=self._krum_threshold,
        )
        logger.info(
            "ChargeShieldAggregator inizializzato — FedAvgAggregator + "
            "PrivacyAuditor + ChargingIDS (fase 2, nessuna DP ancora)"
        )

    # ── Aggregator API ───────────────────────────────────────────────────────

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        """Raccoglie l'update di UN client per il round corrente. Chiamato una
        volta per client da ScatterAndGather non appena il risultato arriva."""
        self._ensure_components()
        import torch
        from ml.base_ml import GradientUpdate

        try:
            dxo = from_shareable(shareable)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"ChargeShieldAggregator.accept: DXO malformato: {exc}", exc_info=True)
            return False

        # VERIFY: assunto dxo.data = dict[str state_dict_key, np.ndarray],
        # stesso formato emesso da chargeshield_executor.py.
        weights_dict: dict[str, Any] = dxo.data or {}
        if not weights_dict:
            logger.warning("ChargeShieldAggregator.accept: DXO senza pesi — scartato")
            return False

        if self._weight_keys is None:
            self._weight_keys = list(weights_dict.keys())

        try:
            ordered_weights = [
                torch.tensor(weights_dict[k]) if not isinstance(weights_dict[k], torch.Tensor)
                else weights_dict[k]
                for k in self._weight_keys
            ]
        except KeyError as exc:
            logger.error(f"ChargeShieldAggregator.accept: chiave mancante {exc} — scartato")
            return False

        # VERIFY: cluster_id preso da dxo.meta (impostato da chargeshield_executor.py),
        # non da fl_ctx/peer info — da confermare che sia il modo corretto/più robusto
        # di identificare il client mittente in un Aggregator NVFLARE.
        node_id = dxo.meta.get("cluster_id", "unknown")
        n_samples = dxo.meta.get("n_samples", 0)
        loss = dxo.meta.get("loss")

        update = GradientUpdate(
            node_id=node_id,
            cluster_id=node_id,
            round_num=self._round_num + 1,  # round che si sta per aggregare
            weights=ordered_weights,
            gradients=None,
            loss=loss,
            n_samples=n_samples,
            metadata={},
        )
        self._round_updates.append(update)
        logger.debug(f"ChargeShieldAggregator.accept: raccolto update da [{node_id}]")
        return True

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """Chiamato da ScatterAndGather dopo che tutti i client (o il timeout)
        hanno risposto per il round corrente. Esegue FedAvg + IDS/Auditor,
        poi restituisce il nuovo modello globale come DXO/Shareable."""
        self._ensure_components()
        import numpy as np

        self._round_num += 1  # VERIFY: leggere il round reale da fl_ctx
        updates = self._round_updates
        self._round_updates = []

        if not updates:
            logger.warning(f"Round {self._round_num}: nessun update ricevuto — aggregazione saltata")
            return DXO(data_kind=DataKind.WEIGHTS, data={}).to_shareable()

        for u in updates:
            self._fedavg.collect(u)
        aggregated = self._fedavg.aggregate(self._round_num)

        # ── IDS/Auditor: replica semplificata di run_ids() per questo round ──
        self._run_ids_analysis(updates)

        if aggregated is None or not aggregated.global_weights:
            logger.error(
                f"Round {self._round_num}: FedAvgAggregator non ha prodotto un "
                "aggregato (partecipanti insufficienti?) — restituisco Shareable vuoto"
            )
            return DXO(data_kind=DataKind.WEIGHTS, data={}).to_shareable()

        # Aggiorna il riferimento per il delta peer-relative del prossimo round.
        # Fase 2 (nessuna DP): il modello aggregato è già "raw" per definizione.
        self._prev_raw_global = aggregated.global_weights

        outgoing_weights = {
            k: (w.detach().cpu().numpy() if hasattr(w, "detach") else np.asarray(w))
            for k, w in zip(self._weight_keys or [], aggregated.global_weights)
        }
        out_dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data=outgoing_weights,
            meta={
                "n_participants": aggregated.n_participants,
                "mean_loss": aggregated.mean_loss,
            },
        )
        return out_dxo.to_shareable()

    # ── IDS/Auditor (replica di run_experiments.py::run_ids(), un round) ─────

    def _run_ids_analysis(self, updates: list[Any]) -> None:
        """
        Stessa logica di run_ids() nella simulazione, ridotta a UN round (qui
        l'aggregatore vede un round alla volta, non l'intero fl_results):
        delta rispetto al modello del round precedente, normalizzazione
        peer-relative (mediana → max_grad_norm), audit per client, Krum sul
        cluster. Vedi scripts/run_experiments.py::run_ids() per la
        spiegazione completa dei fix (Sprint 9) replicati qui.

        Differenza rispetto alla simulazione: qui non esiste ancora un modo
        strutturato per esportare i risultati (niente save_results()/Excel in
        questa fase) — vengono solo loggati. Da estendere quando questa parte
        del progetto arriva a girare davvero.
        """
        import numpy as np
        import torch

        client_deltas: dict[str, list] = {}
        client_norms: dict[str, float] = {}

        for u in updates:
            weights = u.weights or []
            if self._prev_raw_global is not None and len(self._prev_raw_global) == len(weights):
                delta = [
                    (w.float() if isinstance(w, torch.Tensor) else torch.tensor(float(w)))
                    - (g.float() if isinstance(g, torch.Tensor) else torch.tensor(float(g)))
                    for w, g in zip(weights, self._prev_raw_global)
                ]
            else:
                delta = [
                    w.float() if isinstance(w, torch.Tensor) else torch.tensor(float(w))
                    for w in weights
                ]
            l2_sq = sum(float(dw.norm() ** 2) for dw in delta)
            client_deltas[u.node_id] = delta
            client_norms[u.node_id] = float(np.sqrt(max(l2_sq, 1e-12)))

        if client_norms:
            sorted_norms = sorted(client_norms.values())
            median_norm = sorted_norms[(len(sorted_norms) - 1) // 2]
            scale = self._max_grad_norm / median_norm if median_norm >= 1e-4 else 1.0
        else:
            scale = 1.0

        reports: dict[str, Any] = {}
        gradients: dict[str, dict[str, Any]] = {}
        for node_id, delta in client_deltas.items():
            model_update = {f"layer_{i}": dw * scale for i, dw in enumerate(delta)}
            reports[node_id] = self._auditor.audit(
                node_id=node_id, round_id=self._round_num, model_update=model_update,
            )
            gradients[node_id] = {f"layer_{i}": dw for i, dw in enumerate(delta)}

        if not reports:
            return

        analysis = self._ids.analyze_round(self._round_num, reports, gradients)
        if getattr(analysis, "byzantine_nodes", None):
            logger.warning(
                f"Round {self._round_num}: ChargingIDS ha rilevato nodi Byzantine: "
                f"{analysis.byzantine_nodes}"
            )
        else:
            logger.debug(f"Round {self._round_num}: nessun nodo Byzantine rilevato")
