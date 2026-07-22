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

Cosa fa in più da FASE 3 (2026-07-22, pomeriggio) — DP client-side per i modi
che la richiedono (vedi dp_mode sotto e chargeshield_aggregator.py per la
controparte server-side di "dp-fedavg"/"central"):
    - dp_mode="central": il client CLIPPA il proprio update (clip_only(),
      NESSUN rumore) prima di inviarlo — il rumore va sull'aggregato, aggiunto
      dal server in ChargeShieldAggregator.
    - dp_mode="local": il client CLIPPA E RUMORIZZA (privatize()) prima di
      inviare — il server non deve MAI vedere il valore pulito, nemmeno
      transitoriamente (vero local DP).
    - dp_mode="dp-fedavg" (default): il client invia l'update RAW, non
      privatizzato — architetturalmente, in dp-fedavg [McMahan et al. 2017]
      è il SERVER (semi-trusted) a clippare+rumorizzare ogni update non
      appena arriva, PRIMA di aggregarlo — vedi ChargeShieldAggregator.accept().
      Questa è una distinzione che nella simulazione single-process
      (scripts/run_experiments.py) è invisibile (client e server sono la
      stessa chiamata Python), ma che nel deploy NVFLARE reale diventa
      concreta: dp-fedavg e local finiscono per inviare payload diversi sul
      canale di rete (raw vs. già-privatizzato).

Cosa NON fa ancora (fasi successive, vedi docs/NVFlareIntegration.md):
    - Non chiama PrivacyAuditor.audit() / non emette dati per ChargingIDS
      lato client — questa analisi resta server-side in ChargeShieldAggregator,
      mirroring run_ids() (che nella simulazione osserva gli update non
      ancora aggregati, non i singoli client).

Punti da VERIFICARE appena nvflare è installabile (marcati inline con "VERIFY:"):
    - Il formato esatto di dxo.data (dict[str, np.ndarray] atteso, chiavi = nomi
      state_dict) prodotto da FullModelShareableGenerator per un PTFileModelPersistor.
    - Come comunicare n_samples per la pesatura in InTimeAccumulateWeightedAggregator
      (probabilmente DXO.meta[MetaKey.NUM_STEPS_CURRENT_ROUND] o simile — da
      confermare contro la versione 2.7.2 pinnata in Dockerfile.flare).
    - Se il round_num vada letto da fl_ctx (es. via AppConstants.CURRENT_ROUND)
      invece che da un contatore locale (qui uso un contatore locale come
      placeholder, quasi certamente sbagliato in un run multi-round reale).

FIX 2026-07-22 (review indipendente, punto A1 — bug reale nella prima stesura,
non un semplice VERIFY): meta.json deploya un solo app/ a "@ALL" i siti, e
config_fed_client.json ha "cluster_id": "highway" hardcoded — senza correzione,
TUTTI e 4 i client NVFLARE (nvflare/project.yml li nomina "highway", "urban",
"residential", "corporate") avrebbero istanziato cluster_id="highway",
allenandosi tutti sulla STESSA fetta di dati: l'opposto dell'eterogeneità
per-cluster su cui si basa l'intera simulazione. Fix: _setup() deriva ora
cluster_id dal nome del sito NVFLARE (fl_ctx.get_identity_name(), che
project.yml garantisce coincidere con highway/urban/residential/corporate)
quando riconosciuto, usando il valore di config solo come fallback (con
warning) se il nome del sito non corrisponde a nessun cluster noto.
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

# Split contiguo identico a run_fl_rounds() — stesso ordine cluster_ids.
# Definito a livello di modulo (fix 2026-07-22, review A1): usato sia per
# derivare cluster_id dal nome del sito NVFLARE in _setup(), sia per il
# filtro del dataset — prima erano due copie locali della stessa lista.
_CLUSTER_IDS = ["highway", "urban", "residential", "corporate"]


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
        dp_mode: str = "dp-fedavg",
        epsilon: float = 1.0,
        delta: float = 1.0e-5,
        max_grad_norm: float = 1.0,
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

        # DP (fase 3, 2026-07-22) — vedi docstring del modulo per la semantica
        # dei 3 dp_mode. epsilon/delta/max_grad_norm devono combaciare con gli
        # stessi valori usati da ChargeShieldAggregator lato server (per
        # dp_mode="dp-fedavg", dove il server clippa/rumorizza) — nessun
        # meccanismo di validazione incrociata client/server esiste ancora,
        # da tenerli allineati manualmente in config_fed_client.json/
        # config_fed_server.json finché non c'è un modo migliore.
        self._dp_mode = dp_mode
        self._dp_epsilon = epsilon
        self._dp_delta = delta
        self._dp_max_grad_norm = max_grad_norm
        self._gm = None  # GradientManager, istanziato lazy solo se dp_mode ne ha bisogno

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

        site_name = fl_ctx.get_identity_name() if fl_ctx else None

        # Fix 2026-07-22 (review A1): deriva cluster_id dal nome del sito NVFLARE
        # invece di fidarsi ciecamente del valore in config_fed_client.json —
        # nvflare/project.yml nomina i 4 siti client esattamente highway/urban/
        # residential/corporate, e lo stesso config_fed_client.json (con
        # cluster_id="highway" hardcoded) viene deployato a "@ALL" i siti in
        # meta.json. Senza questo, ogni sito userebbe cluster_id="highway".
        if site_name in _CLUSTER_IDS:
            if site_name != self._cluster_id:
                logger.warning(
                    f"cluster_id da config ({self._cluster_id}) diverso dal nome "
                    f"del sito NVFLARE ({site_name}) — uso il nome del sito."
                )
            self._cluster_id = site_name
        else:
            logger.warning(
                f"Nome sito NVFLARE '{site_name}' non riconosciuto come cluster "
                f"({_CLUSTER_IDS}) — uso cluster_id da config: {self._cluster_id}. "
                "Se questo accade su più siti, verranno tutti allenati sugli "
                "stessi dati (vedi FIX 2026-07-22 nel docstring del modulo)."
            )

        node_id = f"{self._cluster_id}-01" if site_name is None else site_name

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

        # Split contiguo identico a run_fl_rounds() — stesso ordine cluster_ids
        # (_CLUSTER_IDS ora definita a livello di modulo, vedi fix 2026-07-22 sopra).
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

        # Riferimento per il clip del DELTA (fix 2026-07-22, stesso principio
        # di run_fl_rounds()/pre_round_weights): i pesi APPENA applicati sopra
        # sono il modello ricevuto a inizio round. Se il server non ha inviato
        # nulla (should not happen in un deploy NVFLARE reale — il persistor
        # inizializza sempre un modello — ma non escluso in questa fase non
        # verificata), usa i pesi correnti del trainer come fallback: assenza
        # di reference fa scattare il fallback storico (clip assoluto) in
        # GradientManager._clip_weights(), non un crash.
        pre_round_weights = self._trainer.get_weights()

        # Training locale — STESSO metodo usato in simulazione da
        # scripts/run_experiments.py::run_fl_rounds().
        update = self._trainer.train_local(self._sessions, self._round_num)

        weight_keys = self._trainer.get_weight_keys()

        # DP client-side (fase 3, 2026-07-22) — vedi docstring del modulo.
        # dp_mode="dp-fedavg": nessuna operazione qui, il client invia l'update
        # raw — è il server (ChargeShieldAggregator.accept()) a clippare e
        # rumorizzare, mirroring l'architettura dp-fedavg originale (McMahan
        # et al. 2017: server semi-trusted riceve il raw update per client).
        if self._dp_mode in ("central", "local"):
            if self._gm is None:
                from ml.gradient_manager import GradientManager
                self._gm = GradientManager({
                    "epsilon": self._dp_epsilon,
                    "delta": self._dp_delta,
                    "max_grad_norm": self._dp_max_grad_norm,
                })
            if self._dp_mode == "central":
                update = self._gm.clip_only(
                    update, weight_keys=weight_keys, reference_weights=pre_round_weights,
                )
            else:  # "local"
                update = self._gm.privatize(
                    update, weight_keys=weight_keys, reference_weights=pre_round_weights,
                )

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
                "dp_mode": self._dp_mode,
                # VERIFY: chiave meta corretta per la pesatura letta da un
                # eventuale futuro aggregatore NVFLARE built-in — irrilevante
                # per ChargeShieldAggregator, che legge "n_samples" sopra
                # direttamente (vedi chargeshield_aggregator.py).
            },
        )
        return outgoing_dxo.to_shareable()
