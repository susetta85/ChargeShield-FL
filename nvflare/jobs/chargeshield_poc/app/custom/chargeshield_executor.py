# nvflare/jobs/chargeshield_poc/app/custom/chargeshield_executor.py
"""
ChargeShieldExecutor — NVFLARE client-side Executor (2026-07-22).

STATO: scritto e ragionato manualmente in un sandbox dove `nvflare` e `torch`
non sono installabili (proxy blocca download.pytorch.org; nvflare non
verificabile senza torch). Vedi docs/NVFlareIntegration.md per lo stato
completo, cosa è verificato solo per lettura del codice sorgente di src/ml/,
e i prossimi passi.

AGGIORNAMENTO (2026-07-24): PRIMO run reale (`make nvflare-sim-smoke`, sulla
macchina dell'utente, nvflare 2.8.1) — ha trovato e permesso di correggere un
bug reale nella risoluzione di _PROJECT_ROOT (vedi commento sotto). Fix
applicato ma non ancora verificato da una riesecuzione riuscita al momento di
questo commit — vedi docs/NVFlareIntegration.md per lo stato aggiornato.

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
      privatizzato — architetturalmente è il SERVER (semi-trusted) a
      clippare+rumorizzare ogni update non appena arriva, PRIMA di aggregarlo
      — vedi ChargeShieldAggregator.accept(). Questo placement (rumore
      per-client prima dell'aggregazione) è una variante più restrittiva e
      non-standard, non descritta letteralmente nell'Algoritmo 1 di McMahan
      et al. 2018 (quel paper corrisponde invece al mode "central").
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

FIX 2026-08-03 (primo job NVFLARE reale completato, poi bloccato provando a
lanciare scripts/run_nvflare_mia.py sul suo dump): _setup() caricava il 100%
delle sessioni disponibili per il training, senza riservare alcun hold-out —
a differenza della simulazione locale (scripts/run_experiments.py::main(),
split 80/20 PRIMA del training FL). Senza un hold-out genuino, l'analisi MIA
offline non ha nessun pool non-member da usare, e scaricare un anno ACN-Data
aggiuntivo si è rivelato non praticabile (nessun anno oltre a quelli già
usati esiste per questi 3 siti). Fix: stesso split seed-based 80/20 della
simulazione, dentro _setup() — vedi commento lì per il dettaglio. Il job già
completato con questo file pre-fix si allenava sul 100% dei dati ed è da
considerare superato, non direttamente confrontabile con run successivi.

AGGIORNAMENTO (2026-08-07): due job NVFLARE reali completati end-to-end sulla
macchina dell'utente (10/10 round, metriche non degeneri e distinte per sito
— vedi experiments/nvflare_ids_audit_results_20260804_133100_58d089.json e il
suo .pkl gemello nvflare_fl_results_20260804_133100_58d089.pkl) confermano
empiricamente il formato dati usato qui (vedi sotto). Il conteggio del round
resta invece un limite noto, non verificabile solo osservando run completati
linearmente — vedi il punto dedicato sotto.

Punti CONFERMATI empiricamente dal job reale del 2026-08-04 (10/10 round
completati, metriche non degeneri — vedi
experiments/nvflare_fl_results_20260804_133100_58d089.pkl), ex-"VERIFY":
    - Il formato esatto di dxo.data (dict[str, np.ndarray], chiavi = nomi
      state_dict) prodotto da FullModelShareableGenerator per un
      PTFileModelPersistor: confermato — se il formato fosse stato sbagliato,
      _sessions_to_tensor()/apply_global_model() avrebbero fallito o prodotto
      NaN/errori invece di 10 round di loss/pesi sensati.

Punto NON verificato (limite noto, non risolvibile osservando run completati
con successo — marcato ancora inline con "VERIFY:"):
    - Se il round_num vada letto da fl_ctx (es. via AppConstants.CURRENT_ROUND)
      invece che da un contatore locale incrementato ad ogni execute(). I job
      completati sono sempre andati linearmente round 1→10 senza interruzioni:
      un contatore locale e un contatore letto da fl_ctx produrrebbero lo
      STESSO valore in quello scenario, quindi il successo dei job non prova
      nulla su questo punto. Fallirebbe silenziosamente (round_num disallineato
      tra client ed effettivo stato server, senza eccezioni) SOLO in scenari
      non ancora testati: resume di un run interrotto, retry di un round
      fallito, o esecuzione di un client che salta round — nessuno di questi
      si è ancora presentato nei job reali eseguiti finora.

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
NOTA (2026-07-22, più tardi lo stesso giorno — trovato da review indipendente
fresh-pass, cosmetico): questi nomi (highway/urban/residential/corporate) sono
storici — narrano il bug così com'era quando fu trovato. Da Sprint 10 in poi
`nvflare/project.yml` nomina i 3 siti reali `caltech`/`jpl`/`office1` (vedi
sezione "3 SITI REALI" più sotto in questo file) — il meccanismo di derivazione
resta identico, solo i nomi concreti sono cambiati.

FIX 2026-07-22 (review indipendente fase 3-5, bug CRITICO trovato in _setup() —
non un VERIFY, un bug funzionale che avrebbe reso l'intero client inutilizzabile):
_setup() caricava le sessioni via ACNDataset.get_sample() e le usava
DIRETTAMENTE, senza mai applicare l'equivalente di enrich_sessions()/
normalize_sessions() (scripts/run_experiments.py). AutoencoderTrainer.
CONTINUOUS_FEATURES include "hour_of_day"/"duration_hours", calcolati SOLO da
enrich_sessions() — assenti nei sample grezzi. Senza il fix, _sessions_to_tensor()
(src/ml/autoencoder_trainer.py) scartava OGNI sessione per feature mancante,
producendo un tensore vuoto: nessun training locale sarebbe mai realmente
avvenuto in un run NVFLARE reale, un bug invisibile a py_compile e non
menzionato nei "VERIFY:" originali (che riguardavano solo il formato dei pesi,
non i dati di input). Fix: _enrich_sessions()/_compute_feature_stats()/
_normalize_sessions() (duplicati da scripts/run_experiments.py — non importati,
per non riconfigurare logging.basicConfig() dentro un processo client NVFLARE
reale) chiamati in _setup() prima dello split per-cluster.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any

# ── Path setup: rende importabile src/ (stesso layout di scripts/run_experiments.py) ──
# BUG REALE trovato al primo run vero (2026-07-24, `make nvflare-sim-smoke`,
# prima esecuzione in assoluto di questo file): `Path(__file__).resolve().
# parents[4]` assumeva che questo file restasse per sempre a
# nvflare/jobs/chargeshield_poc/app/custom/ (dove parents[4] è davvero la
# project root). Ma `nvflare simulator` COPIA custom/ dentro il workspace
# (osservato: nvflare/sim_workspace/<client>/simulate_job/app_<client>/custom/,
# una profondità diversa) — da lì, parents[4] risolve dentro sim_workspace/,
# non la project root. Effetto reale: "Directory dataset non trovata:
# .../sim_workspace/datasets/acn/caltech" — ogni client avrebbe caricato 0
# sessioni, esattamente il tipo di fallimento silenzioso già temuto (qui però
# loggato ad ERROR, non silenzioso — solo scoperto perché qualcuno ha letto i
# log, non perché il codice l'ha impedito).
#
# Fix: _find_project_root() risolve la project root in modo indipendente da
# dove NVFLARE fisicamente copia questo file — priorità a una variabile
# d'ambiente esplicita (settata da `make nvflare-sim*`), poi risalita da
# __file__ cercando pyproject.toml di chargeshield-fl come fallback per chi
# lancia nvflare manualmente senza passare dal Makefile.
def _find_project_root() -> Path:
    env_root = __import__("os").environ.get("CHARGESHIELD_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file():
            try:
                if 'name = "chargeshield-fl"' in pyproject.read_text():
                    return candidate
            except OSError:
                continue
    raise RuntimeError(
        "Impossibile trovare la project root di ChargeShield-FL: imposta "
        "CHARGESHIELD_PROJECT_ROOT nell'ambiente, oppure esegui tramite "
        "'make nvflare-sim'/'make nvflare-sim-smoke' (che la impostano già)."
    )


_PROJECT_ROOT = _find_project_root()  # .../ChargeShield-FL
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

# I 3 SITI REALI di ACN-Data (2026-07-22, sostituisce i 4 nomi fittizi
# highway/urban/residential/corporate — mai stati siti realmente distinti,
# solo fette arbitrarie dello stesso dataset). Verificati via siteID +
# conteggio stazioni uniche contro https://ev.caltech.edu/dataset (Caltech 54
# EVSE, JPL 50 EVSE, Office 1 8 EVSE) — vedi scripts/run_experiments.py per lo
# stesso identico controllo lato simulazione, e docs/NVFlareIntegration.md per
# il dettaglio della scoperta (il vecchio dataset "jpl" del progetto era in
# realtà Caltech). Nessun 4°/5° sito reale esiste in questo dataset — i 2
# client sintetici usati per la validazione IDS a 5 client (vedi
# config/experiment.yaml) esistono SOLO nella simulazione, non ancora portati
# su NVFLARE (fuori scope per ora, vedi nvflare/project.yml).
_CLUSTER_IDS = ["caltech", "jpl", "office1"]

# ── Enrichment/normalizzazione sessioni — duplicati da scripts/run_experiments.py ──
# FIX 2026-07-22 (review indipendente fase 3-5, bug CRITICO trovato in _setup()):
# la prima stesura caricava le sessioni via ACNDataset.get_sample() e le passava
# DIRETTAMENTE ad AutoencoderTrainer.train_local(), senza mai chiamare
# l'equivalente di enrich_sessions()/normalize_sessions() (scripts/run_experiments.py).
# AutoencoderTrainer.CONTINUOUS_FEATURES include "hour_of_day" e "duration_hours",
# che NON esistono nei sample grezzi di ACNDataset (solo start_time/end_time in
# formato ISO) — vengono calcolati SOLO da enrich_sessions(). Senza questo fix,
# _sessions_to_tensor() scarta OGNI sessione (val is None → valid=False per ogni
# riga, vedi src/ml/autoencoder_trainer.py::_sessions_to_tensor()), quindi
# self._sessions avrebbe prodotto un tensore vuoto e train_local() non avrebbe
# mai potuto addestrare nulla — un bug che avrebbe reso l'intero client NVFLARE
# non funzionante su dati reali, non rilevabile da py_compile.
#
# Duplicate qui (non importate da scripts/run_experiments.py) perché quel modulo
# chiama logging.basicConfig() a livello di modulo — importarlo da dentro un
# processo client NVFLARE reale riconfigurerebbe silenziosamente il logging
# dell'intero processo.
#
# BUG REALE trovato da review indipendente (2026-07-24, round successivo al
# primo run reale): questa funzione NON era "stessa formula esatta" come
# dichiarato — mancava del tutto il fix timezone del 2026-07-22 applicato al
# vero enrich_sessions() in scripts/run_experiments.py (vedi quella funzione:
# ACN-Data porta un suffisso "GMT" fuorviante, connectionTime/disconnectTime
# sono in realtà UTC, e hour_of_day va calcolato sull'ora LOCALE del sito via
# il campo "timezone" di ogni sessione, non su start.hour grezzo — altrimenti
# sfasato di 7-8h da quanto la feature dichiara di rappresentare). La versione
# NVFLARE usava ancora `float(start.hour)` diretto (comportamento pre-fix):
# ogni client NVFLARE avrebbe addestrato su un hour_of_day sistematicamente
# sbagliato, mai confrontabile con i risultati della simulazione. Non
# catturato dal controllo empirico "2652/2652 sessioni valide" del
# 2026-07-22 (docs/NVFlareIntegration.md) perché quel controllo verifica solo
# che le feature esistano e siano nel range [0,1] atteso dopo normalizzazione
# — un offset costante ma sbagliato supera comunque quel controllo. Fix:
# stessa identica logica ZoneInfo di scripts/run_experiments.py::
# enrich_sessions(), portata qui parola per parola.
def _enrich_sessions(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggiunge hour_of_day/duration_hours dai timestamp — stessa logica di
    scripts/run_experiments.py::enrich_sessions() (localizzazione timezone
    inclusa, fix 2026-07-24)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    enriched = []
    for s in sessions:
        try:
            start = datetime.fromisoformat(s["start_time"])
            end = datetime.fromisoformat(s["end_time"])

            tz_name = s.get("timezone")
            if tz_name:
                try:
                    local_start = start.replace(tzinfo=ZoneInfo("UTC")).astimezone(
                        ZoneInfo(tz_name)
                    )
                    hour_of_day = float(local_start.hour)
                except Exception:
                    # Timezone IANA sconosciuta/malformata — fallback all'ora
                    # grezza invece di scartare la sessione (stesso principio
                    # dell'originale: un hour_of_day leggermente sfasato è
                    # preferibile a perdere il campione).
                    hour_of_day = float(start.hour)
            else:
                hour_of_day = float(start.hour)  # nessun timezone noto — fallback

            s["hour_of_day"] = hour_of_day
            s["duration_hours"] = max(0.0, (end - start).total_seconds() / 3600.0)
            enriched.append(s)
        except (KeyError, ValueError):
            pass  # scarta sessioni con timestamp malformati (stesso comportamento dell'originale)
    return enriched


def _compute_feature_stats(
    sessions: list[dict[str, Any]], features: list[str],
) -> dict[str, tuple[float, float]]:
    """Min/max per-feature — vedi scripts/run_experiments.py::compute_feature_stats().
    Calcolato su TUTTE le sessioni del dataset condiviso (fase 1: un solo file
    per tutti i client — vedi limitazione "Per-client dataset access è fake" in
    docs/NVFlareIntegration.md), non solo sulla fetta di questo cluster: un vero
    client mono-sito non potrebbe calcolare min/max globali da solo, ma finché
    il dataset resta condiviso, farlo qui (prima dello split) è l'equivalente
    più fedele possibile di come lo fa la simulazione (compute_feature_stats su
    train_sessions, che copre già tutti i cluster nello stesso processo)."""
    stats: dict[str, tuple[float, float]] = {}
    for feat in features:
        vals = []
        for s in sessions:
            raw = s.get(feat)
            if raw is None:
                continue
            try:
                vals.append(float(raw))
            except (ValueError, TypeError):
                continue
        if not vals:
            stats[feat] = (0.0, 1.0)
            continue
        fmin, fmax = min(vals), max(vals)
        stats[feat] = (fmin, fmax if fmax != fmin else fmin + 1.0)
    return stats


def _normalize_sessions(
    sessions: list[dict[str, Any]], stats: dict[str, tuple[float, float]], features: list[str],
) -> list[dict[str, Any]]:
    """Min-max scaling [0,1] — vedi scripts/run_experiments.py::normalize_sessions()."""
    normalized = []
    for s in sessions:
        s = dict(s)
        for feat in features:
            val = s.get(feat)
            if val is None:
                continue
            fmin, fmax = stats[feat]
            s[feat] = (float(val) - fmin) / (fmax - fmin)
        normalized.append(s)
    return normalized


class ChargeShieldExecutor(Executor):
    """
    Avvolge AutoencoderTrainer (src/ml/autoencoder_trainer.py) per l'esecuzione
    reale su un client NVFLARE, invece della simulazione single-process di
    scripts/run_experiments.py::run_fl_rounds().

    Args (da config_fed_client.json):
        cluster_id:   uno tra caltech/jpl/office1 (2026-07-22: siti REALI,
                      non più nomi fittizi) — determina quale sottocartella
                      datasets/acn/<cluster_id>/ questo client carica. Ogni
                      sito ha ORA davvero solo i propri dati (tutti gli anni
                      disponibili in quella sottocartella), non una fetta
                      arbitraria di un file condiviso — vedi _setup().
        input_dim/lr/epochs/batch_size/proximal_mu/seed: passati direttamente
                      alla config di AutoencoderTrainer, stessi nomi/semantica
                      di scripts/run_experiments.py.
        dataset_path: directory PADRE dei dataset per sito (default:
                      "datasets/acn") — _setup() vi accoda self._cluster_id
                      per ottenere la cartella reale del sito (es.
                      "datasets/acn/caltech/") e carica TUTTI i file .json
                      al suo interno (tutti gli anni disponibili).
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
        dataset_path: str = "datasets/acn",
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
        # LIMITE NOTO (non un VERIFY generico): contatore locale, mai confrontato
        # con AppConstants.CURRENT_ROUND da fl_ctx. I 2 job NVFLARE reali completati
        # (10/10 round, sempre lineari 1→10 senza interruzioni) non possono
        # confermare né smentire questo punto — un contatore locale e uno letto da
        # fl_ctx producono lo stesso valore in un run lineare. Resterebbe
        # silenziosamente disallineato solo in scenari mai testati: resume di un
        # run interrotto, retry di un round fallito, client che salta un round.
        self._round_num = 0

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
        # nvflare/project.yml nomina i siti client (oggi: caltech/jpl/office1,
        # i 3 siti reali ACN-Data — storicamente erano highway/urban/
        # residential/corporate, vedi nota in cima al file), e lo stesso
        # config_fed_client.json (con cluster_id hardcoded a un solo sito)
        # viene deployato a "@ALL" i siti in meta.json. Senza questo, ogni
        # sito userebbe lo stesso cluster_id hardcoded in config.
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

        # FIX 2026-07-22 (3 siti reali, sostituisce fase 1): ogni client carica
        # ORA il proprio dataset REALE — tutti i file .json nella sua directory
        # datasets/acn/<cluster_id>/ (es. datasets/acn/caltech/, un file per
        # anno) — non più un unico file condiviso affettato per indice tra 4
        # cluster fittizi. dataset_path (da config_fed_client.json) è ora la
        # directory PADRE ("datasets/acn"), non un singolo file; self._cluster_id
        # (derivato dal nome del sito NVFLARE sopra) seleziona la sottocartella.
        # Risolve la limitazione "Per-client dataset access è fake" documentata
        # in docs/NVFlareIntegration.md — ogni sito ha ORA davvero solo i propri
        # dati, non una fetta arbitraria di un pool condiviso.
        dataset_dir = _PROJECT_ROOT / self._dataset_path / self._cluster_id
        if not dataset_dir.is_dir():
            logger.error(f"[{self._cluster_id}] Directory dataset non trovata: {dataset_dir}")
            self._sessions = []
            return

        all_sessions: list[dict[str, Any]] = []
        json_files = sorted(dataset_dir.glob("*.json"))
        for f in json_files:
            ds = ACNDataset()
            ds.load(str(f))
            all_sessions.extend(ds.get_sample(i) for i in range(len(ds)))

        if not all_sessions:
            logger.error(f"[{self._cluster_id}] Nessuna sessione trovata in {dataset_dir}")
            self._sessions = []
            return

        # FIX 2026-07-22 (review indipendente, bug critico — vedi commento sopra
        # _enrich_sessions()): enrich PRIMA della normalizzazione, così
        # hour_of_day/duration_hours esistono per ogni sessione prima che
        # _sessions_to_tensor() le richieda. Senza questo, ogni sessione veniva
        # scartata silenziosamente e self._sessions produceva un tensore vuoto.
        all_sessions = _enrich_sessions(all_sessions)

        # FIX 2026-08-03 (bug reale trovato provando a lanciare per la prima
        # volta scripts/run_nvflare_mia.py sul dump del primo job NVFLARE
        # completato: ogni sito aveva già usato TUTTI gli anni scaricati per
        # il training — datasets/acn/<sito>/ non ha alcun anno "mai visto",
        # quindi nessun hold-out genuino esisteva per calcolare gli AUC di
        # membership inference. Scaricare un anno in più (es. 2022) non era
        # praticabile: verificato che ACN-Data non ne ha per questi 3 siti).
        # Fix strutturale, non un ripiego: split seed-based 80/20 PRIMA del
        # training, stessa identica logica di scripts/run_experiments.py::
        # main() (random.seed(seed); random.shuffle(sessions); split
        # all'80%) — invece che un secondo dataset esterno, ogni sito
        # riserva ora il 20% delle proprie sessioni come hold-out, mai
        # passato a train_local(). scripts/run_nvflare_mia.py::
        # load_client_sessions() ricostruisce lo STESSO split (stesso seed,
        # stesso ordine di caricamento file, stessa enrichment) per
        # ottenere il 20% hold-out senza bisogno di alcun file esterno —
        # zero data leakage, stessa garanzia anti-leakage della simulazione
        # locale. Invalida il primo job NVFLARE completato (si era allenato
        # sul 100% dei dati, non più confrontabile con questo split) — va
        # ripetuto (solo submit_job, nessun rebuild: questo file non è
        # incluso nell'immagine Docker, viene ridistribuito ad ogni
        # sottomissione del job).
        seed = self._trainer_cfg.get("seed", 42)
        random.seed(seed)
        random.shuffle(all_sessions)
        split = max(1, int(len(all_sessions) * 0.8))
        train_sessions = all_sessions[:split]
        n_holdout = len(all_sessions) - len(train_sessions)

        # Normalizzazione [0,1]: calcolata SOLO su train_sessions (no leakage
        # dall'hold-out riservato sopra — stesso principio di
        # scripts/run_experiments.py::main()). Calcolata sulle sessioni DI
        # QUESTO SITO (non più su un pool condiviso multi-cluster, dato che
        # ogni client ora ha davvero solo i propri dati). NOTA/limite noto:
        # questo significa min/max leggermente diversi tra siti (es. il kWh
        # massimo osservato a Caltech vs JPL) invece di un'unica scala
        # globale condivisa come nella simulazione (scripts/run_experiments.py
        # ::compute_feature_stats() su train_sessions di TUTTI i siti
        # insieme) — una differenza reale tra i due percorsi codice,
        # accettabile per un primo draft (i range delle feature ACN-Data non
        # variano di ordini di grandezza tra siti), ma da tenere presente se
        # si confrontano risultati NVFLARE vs simulazione.
        feature_stats = _compute_feature_stats(train_sessions, AutoencoderTrainer.CONTINUOUS_FEATURES)
        self._sessions = _normalize_sessions(
            train_sessions, feature_stats, AutoencoderTrainer.CONTINUOUS_FEATURES
        )

        logger.info(
            f"[{self._cluster_id}] ChargeShieldExecutor pronto — "
            f"{len(self._sessions)} sessioni di training (hold-out riservato: "
            f"{n_holdout}) caricate da {len(json_files)} file in {dataset_dir}"
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

        # CONFERMATO empiricamente dal job reale del 2026-08-04 (10/10 round
        # completati, metriche non degeneri — vedi
        # experiments/nvflare_fl_results_20260804_133100_58d089.pkl): dxo.data è
        # dict[str state_dict_key, np.ndarray], prodotto da
        # FullModelShareableGenerator a partire dal modello persistito da
        # PTFileModelPersistor — se il formato fosse stato diverso (es. tensori
        # invece di ndarray, o chiavi diverse), get_weight_keys()/apply_global_model()
        # sotto avrebbero fallito con KeyError o crash invece di completare 10 round.
        global_weights_dict: dict[str, Any] = incoming_dxo.data or {}

        # LIMITE NOTO (vedi commento su self._round_num nell'__init__): contatore
        # locale mai confrontato con fl_ctx. Non testato in scenari di resume/
        # retry/round-skip — solo run lineari 1→10 finora.
        self._round_num += 1

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
        # rumorizzare — placement per-client non-standard, non descritto
        # letteralmente nell'Algoritmo 1 di McMahan et al. 2018 (quel paper
        # corrisponde invece al mode "central": clip lato client + un solo
        # rumore lato server sull'aggregato).
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
                # NON APPLICABILE ai job reali eseguiti finora (non un "VERIFY"
                # confermabile da essi): la chiave "n_samples" qui è quella
                # effettivamente letta da ChargeShieldAggregator.accept()
                # (dxo.meta.get("n_samples", 0), vedi chargeshield_aggregator.py)
                # — questo è confermato dai job reali. Resta invece non
                # verificato (e non esercitato dai job reali, che usano sempre
                # il ChargeShieldAggregator custom) se questa stessa chiave
                # sarebbe quella giusta per un ipotetico futuro aggregatore
                # built-in NVFLARE (es. InTimeAccumulateWeightedAggregator, che
                # si aspetta probabilmente DXO.meta[MetaKey.NUM_STEPS_CURRENT_ROUND]
                # o simile) — irrilevante finché non si sostituisce
                # ChargeShieldAggregator con un aggregatore built-in.
            },
        )
        return outgoing_dxo.to_shareable()
