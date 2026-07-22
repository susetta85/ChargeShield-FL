# nvflare/jobs/chargeshield_poc/app/custom/chargeshield_executor.py
"""
ChargeShieldExecutor — NVFLARE client-side Executor (2026-07-22).

STATO: scritto e ragionato manualmente in un sandbox dove `nvflare` e `torch`
non sono installabili (proxy blocca download.pytorch.org; nvflare non
verificabile senza torch). NON eseguito, NON testato. Vedi
docs/NVFlareIntegration.md per lo stato completo, cosa è verificato solo per
lettura del codice sorgente di src/ml/, e i prossimi passi.

Cosa fa (fase 1 — SOLO round-trip, nessun DP/IDS ancora):
    1. Riceve il modello globale dal server (Shareable/DXO).
    2. Lo carica in un AutoencoderTrainer (la STESSA classe usata da
       scripts/run_experiments.py in simulazione single-process — src/ml/
       autoencoder_trainer.py — non una re-implementazione).
    3. Esegue AutoencoderTrainer.train_local() sulle sessioni locali del
       cluster di questo client.
    4. Restituisce l'update al server come Shareable/DXO.

Cosa NON fa ancora (fasi successive, vedi docs/NVFlareIntegration.md):
    - Non applica GradientManager.privatize()/clip_only() (DP) — fase 2.
    - Non chiama PrivacyAuditor.audit() / non emette dati per ChargingIDS — fase 2.
    - Il server usa l'aggregatore built-in di NVFLARE (InTimeAccumulateWeightedAggregator),
      non FedAvgAggregator — sostituirlo con un Aggregator custom che richiami
      FedAvgAggregator è fase 2, per riusare l'accumulo/pesatura già testato.

Punti da VERIFICARE appena nvflare è installabile (marcati inline con "VERIFY:"):
    - Il formato esatto di dxo.data (dict[str, np.ndarray] atteso, chiavi = nomi
      state_dict) prodotto da FullModelShareableGenerator per un PTFileModelPersistor.
    - Come comunicare n_samples per la pesatura in InTimeAccumulateWeightedAggregator
      (probabilmente DXO.meta[MetaKey.NUM_STEPS_CURRENT_ROUND] o simile — da
      confermare contro la versione 2.7.2 pinnata in Dockerfile.flare).
    - Se il round_num vada letto da fl_ctx (es. via AppConstants.CURRENT_ROUND)
      invece che da un contatore locale (qui uso un contatore locale come
      placeholder, quasi certamente sbagliato in un run multi-round reale).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# ── Path setup: rende importabile src/ (stesso layout di scripts/run_experiments.py) ──
# VERIFY: in un vero deploy NVFLARE, src/ deve essere sul PYTHONPATH del processo
# client (es. copiato dentro custom/, o PYTHONPATH impostato nello startup kit
# generato da `nvflare provision`). Qui si assume che il repo sia montato allo
# stesso path relativo usato da scripts/run_experiments.py.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # .../ChargeShield-FL
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nvflare.apis.dxo import DXO, DataKind, from_shareable  # noqa: E402
from nvflare.apis.event_type import EventType  # noqa: E402
from nvflare.apis.executor import Executor  # noqa: E402
from nvflare.apis.fl_constant import ReturnCode  # noqa: E402
from nvflare.apis.fl_context import FLContext  # noqa: E402
from nvflare.apis.shareable import Shareable, make_reply  # noqa: E402
from nvflare.apis.signal import Signal  # noqa: E402

logger = logging.getLogger(__name__)


class ChargeShieldExecutor(Executor):
    """
    Avvolge AutoencoderTrainer (src/ml/autoencoder_trainer.py) per l'esecuzione
    reale su un client NVFLARE, invece della simulazione single-process di
    scripts/run_experiments.py::run_fl_rounds().

    Args (da config_fed_client.json):
        cluster_id:   uno tra highway/urban/residential/corporate — determina
                      quali sessioni carica questo client (oggi: filtro su
                      un dataset locale; in produzione ogni client avrebbe il
                      proprio file/DB locale, non un filtro su un file condiviso).
        input_dim/lr/epochs/batch_size/proximal_mu/seed: passati direttamente
                      alla config di AutoencoderTrainer, stessi nomi/semantica
                      di scripts/run_experiments.py.
        dataset_path: path al JSON ACN-Data da cui questo client carica le
                      proprie sessioni (fase 1: stesso dataset condiviso,
                      filtrato client-side — NON rappresentativo di un vero
                      deploy multi-sito, dove ogni client avrebbe già solo i
                      propri dati; sufficiente per validare il round-trip).
    """

    def __init__(
        self,
        cluster_id: str,
        input_dim: int = 6,
        lr: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        proximal_mu: float = 0.0,
        seed: int = 42,
        dataset_path: str = "datasets/acn/jpl/acndata_sessions_2019.json",
        train_task_name: str = "train",
    ):
        super().__init__()
        self._cluster_id = cluster_id
        self._trainer_cfg = {
            "input_dim": input_dim,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "proximal_mu": proximal_mu,
            "seed": seed,
        }
        self._dataset_path = dataset_path
        self._train_task_name = train_task_name

        self._trainer = None          # AutoencoderTrainer, creato in _setup()
        self._sessions: list[dict[str, Any]] = []
        self._round_num = 0           # VERIFY: leggere da fl_ctx invece di contare localmente

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def handle_event(self, event_type: str, fl_ctx: FLContext) -> None:
        if event_type == EventType.START_RUN:
            self._setup(fl_ctx)

    def _setup(self, fl_ctx: FLContext) -> None:
        """Istanzia AutoencoderTrainer e carica le sessioni del cluster — una
        volta sola per run, non ad ogni round (lo stato del trainer, ottimizzatore
        incluso, deve persistere tra round — vedi ricognizione in
        docs/NVFlareIntegration.md sullo stato persistente richiesto)."""
        from adapters.acn_dataset import ACNDataset
        from ml.autoencoder_trainer import AutoencoderTrainer

        node_id = fl_ctx.get_identity_name() if fl_ctx else f"{self._cluster_id}-01"

        self._trainer = AutoencoderTrainer(
            config=self._trainer_cfg,
            node_id=node_id,
            cluster_id=self._cluster_id,
        )

        # Carica e filtra le sessioni per questo cluster.
        # FASE 1 (round-trip only): filtro client-side su un dataset condiviso,
        # con lo stesso split contiguo per-cluster di run_fl_rounds() — NON
        # rappresenta un vero deploy multi-sito (ogni stazione avrebbe già solo
        # i propri dati). Sufficiente per validare che il trasporto funzioni.
        dataset_path = _PROJECT_ROOT / self._dataset_path
        if not dataset_path.exists():
            logger.error(f"[{self._cluster_id}] Dataset non trovato: {dataset_path}")
            self._sessions = []
            return

        ds = ACNDataset()
        ds.load(str(dataset_path))
        all_sessions = [ds.get_sample(i) for i in range(len(ds))]

        # Split contiguo identico a run_fl_rounds() — stesso ordine cluster_ids.
        _CLUSTER_IDS = ["highway", "urban", "residential", "corporate"]
        if self._cluster_id not in _CLUSTER_IDS:
            logger.error(f"cluster_id sconosciuto: {self._cluster_id}")
            self._sessions = []
            return
        idx = _CLUSTER_IDS.index(self._cluster_id)
        cluster_size = max(1, len(all_sessions) // len(_CLUSTER_IDS))
        start = idx * cluster_size
        end = None if idx == len(_CLUSTER_IDS) - 1 else start + cluster_size
        self._sessions = all_sessions[start:end]

        logger.info(
            f"[{self._cluster_id}] ChargeShieldExecutor pronto — "
            f"{len(self._sessions)} sessioni caricate da {dataset_path.name}"
        )

    # ── Task execution ───────────────────────────────────────────────────────

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        if task_name != self._train_task_name:
            return make_reply(ReturnCode.TASK_UNKNOWN)

        if self._trainer is None:
            logger.error(f"[{self._cluster_id}] Trainer non inizializzato — START_RUN mancato?")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        try:
            incoming_dxo = from_shareable(shareable)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[{self._cluster_id}] DXO malformato: {exc}", exc_info=True)
            return make_reply(ReturnCode.BAD_TASK_DATA)

        # VERIFY: assunto dxo.data = dict[str state_dict_key, np.ndarray],
        # prodotto da FullModelShareableGenerator a partire dal modello
        # persistito da PTFileModelPersistor. Se il formato reale differisce
        # (es. tensori invece di ndarray, o chiavi diverse), questa conversione
        # va adattata.
        global_weights_dict: dict[str, Any] = incoming_dxo.data or {}

        self._round_num += 1  # VERIFY: leggere il round reale da fl_ctx, non contare localmente

        if global_weights_dict:
            weight_keys = self._trainer.get_weight_keys()
            try:
                ordered_weights = [global_weights_dict[k] for k in weight_keys]
            except KeyError as exc:
                logger.error(
                    f"[{self._cluster_id}] Chiave mancante nel modello globale "
                    f"ricevuto: {exc}. Chiavi attese (state_dict): {weight_keys}",
                )
                return make_reply(ReturnCode.BAD_TASK_DATA)
            # AggregatedUpdate è un dataclass di base_ml — costruito qui al volo
            # solo per riusare AutoencoderTrainer.apply_global_model() invariata.
            from ml.base_ml import AggregatedUpdate
            self._trainer.apply_global_model(
                AggregatedUpdate(
                    round_num=self._round_num,
                    global_weights=ordered_weights,
                    n_participants=0,   # non usato da apply_global_model()
                    mean_loss=None,
                )
            )

        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Training locale — STESSO metodo usato in simulazione da
        # scripts/run_experiments.py::run_fl_rounds(). Nessun DP applicato in
        # questa fase (vedi docstring del modulo).
        update = self._trainer.train_local(self._sessions, self._round_num)

        weight_keys = self._trainer.get_weight_keys()
        outgoing_weights = {
            k: w.detach().cpu().numpy() if hasattr(w, "detach") else w
            for k, w in zip(weight_keys, update.weights or [])
        }

        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data=outgoing_weights,
            meta={
                "n_samples": update.n_samples or 0,
                "loss": update.loss,
                "cluster_id": self._cluster_id,
                # VERIFY: chiave meta corretta per la pesatura letta da
                # InTimeAccumulateWeightedAggregator — probabilmente serve
                # anche/invece MetaKey.NUM_STEPS_CURRENT_ROUND
                # (nvflare.apis.fl_constant.MetaKey), da confermare.
            },
        )
        return outgoing_dxo.to_shareable()
