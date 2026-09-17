# nvflare/jobs/chargeshield_poc/app/custom/chargeshield_aggregator.py
"""
ChargeShieldAggregator — NVFLARE server-side Aggregator custom (fase 2+3, 2026-07-22).

STATO: scritto e ragionato manualmente in un sandbox dove `nvflare`/`torch`
non sono installabili (stesso limite di chargeshield_executor.py — vedi
docs/NVFlareIntegration.md).

AGGIORNAMENTO (2026-07-24): PRIMO run reale (`make nvflare-sim-smoke`, sulla
macchina dell'utente, nvflare 2.8.1) — ha trovato e permesso di correggere due
bug reali: (1) risoluzione di _PROJECT_ROOT rotta dal modo in cui `nvflare
simulator` copia questo file nel workspace (vedi commento sotto); (2)
_ensure_components() trattava un fallimento parziale come "già inizializzato"
(vedi commento su self._components_ready nell'__init__). Fix applicati ma non
ancora verificati da una riesecuzione riuscita al momento di questo commit —
vedi docs/NVFlareIntegration.md per lo stato aggiornato.

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
          client + ByzantineDetector.analyze_round() sul round completo — con la
          STESSA logica di normalizzazione peer-relative (mediana) e lo
          stesso calcolo del delta rispetto al modello del round precedente
          usati in run_experiments.py::run_ids(). Vedi quella funzione per la
          spiegazione completa (fix Sprint 9: GRADIENT_EXPLOSION, Krum).
       c) converte l'AggregatedUpdate risultante in un DXO/Shareable per
          FullModelShareableGenerator.

Cosa fa in più da FASE 3 (2026-07-22, pomeriggio) — controparte server-side
della DP (vedi chargeshield_executor.py per la parte client-side):
    - dp_mode="dp-fedavg": gli update ricevuti in accept() sono RAW (il client
      non applica DP) — aggregate() li usa RAW per l'analisi IDS/Auditor
      (mirroring "il server vede transitoriamente il raw update" di run_ids()),
      poi li clippa+rumorizza UNO PER UNO (GradientManager.privatize()) PRIMA
      di passarli a FedAvgAggregator. Questo è un placement per-client
      non-standard: il server (semi-trusted) fa il lavoro di privatizzazione,
      non il client — non è ciò che descrive letteralmente l'Algoritmo 1 di
      McMahan et al. 2018 (vedi "central" sotto per il mode a cui quel paper
      corrisponde davvero).
    - dp_mode="central" [McMahan et al. 2018 — meccanismo esatto del loro
      Algoritmo 1 (DP-FedAvg): clip lato client, un solo draw di rumore lato
      server sull'aggregato]: gli update ricevuti sono già clippati (non
      rumorizzati) dal client — l'IDS li analizza così com'è, FedAvgAggregator
      li combina, poi aggregate() aggiunge UN SOLO rumore Gaussiano
      all'aggregato (GradientManager.privatize_aggregate()), mirroring
      run_fl_rounds().
    - dp_mode="local": gli update ricevuti sono già clippati E rumorizzati dal
      client — l'aggregatore non fa nulla in più per la DP, ma l'IDS ora
      analizza dati già rumorizzati invece che puliti: stessa degradazione
      attesa e documentata in run_ids() (fallback a confronto su pesi
      assoluti/rumorizzati, non un bug).

Cosa fa in più da FASE 4 (2026-07-22, sera) — export strutturato:
    - _run_ids_analysis() ora costruisce un dizionario per-round nello STESSO
      formato di ids_results in scripts/run_experiments.py::run_ids() (alerts
      con node_id/severity/reasons/recommended_action, byzantine_detected,
      low_similarity_nodes) PIU' un blocco per_client_audit con l'output grezzo
      di PrivacyAuditor.audit() (privacy_score, epsilon, threats_detected) —
      stessa struttura dati, non una re-invenzione.
    - _export_results() scrive l'intera cronologia (self._audit_history) su
      un file JSON dopo OGNI round (overwrite, non append — così il file è
      sempre coerente anche se il job viene interrotto a metà), invece di
      solo loggare. Path di default: experiments/nvflare_ids_audit_results.json
      (stessa directory `experiments/` già usata — e già in .gitignore — per
      gli output della simulazione single-process), configurabile via
      `results_export_path`.
    - Non ancora fatto: nessuna analisi MIA (LiRA/Shadow/Yeom) qui — quella è
      fase 5 (raw-update extraction per gli attacchi), non fase 4.

Cosa fa in più da FASE 5 (2026-07-22, notte) — raw-update extraction per LiRA/Shadow:
    - DECISIONE DI DESIGN: LiRA (scripts/run_experiments.py::run_lira()) è già,
      anche nella simulazione, un'analisi POST-HOC che itera sull'intero dict
      fl_results DOPO che tutti i round sono finiti — non un componente che
      gira "dentro" il training loop. run_lira() ha richiesto CINQUE round di
      fix empirici (vedi la sua docstring) trovati eseguendo davvero il codice
      su dati reali; riscriverlo "alla cieca" per girare dentro aggregate()
      (senza poter eseguire nulla in questo sandbox — niente torch/nvflare)
      sarebbe con altissima probabilità un secondo tentativo silenziosamente
      rotto. Scelta fatta invece: l'Aggregator si limita a esportare, per ogni
      round, ESATTAMENTE la stessa struttura dati che run_fl_rounds() produce
      in memoria per la simulazione (stessi 5 campi: mean_loss, n_participants,
      updates, raw_updates, raw_global_weights, global_weights — vedi
      run_fl_rounds() per il contratto esatto). Un nuovo script separato,
      scripts/run_nvflare_mia.py, carica questo dump e chiama run_lira()/
      run_ids()/run_yeom()/save_results() SENZA MODIFICARLI — zero rischio
      di introdurre bug nuovi nella logica di attacco già validata.
    - _fl_results_history[round_num] viene costruito in aggregate() con:
        "raw_updates":        received_updates (i GradientUpdate così come
                               arrivati in accept(), PRIMA di qualunque DP
                               server-side) se dp_mode != "local", altrimenti
                               None — stessa semantica di _store_raw in
                               run_fl_rounds() (sotto local DP il server non
                               deve mai vedere nulla di meno rumoroso di
                               quello che i client hanno già inviato).
        "raw_global_weights": media pesata (per n_samples) dei soli
                               raw_updates, quando non-None — stessa formula
                               di run_fl_rounds(), usata da run_ids() come
                               riferimento pulito per il delta peer-relative.
        "updates":             updates_for_fedavg (ciò che è stato REALMENTE
                               passato a FedAvgAggregator — post-privatize
                               server-side in dp-fedavg, as-received in
                               central/local).
        "global_weights":      aggregated.global_weights DOPO l'eventuale
                               rumore central-DP sull'aggregato — identico
                               a ciò che viene ridistribuito ai client.
      Nota su "central": qui received_updates sono GIA' clippati (fatto dal
      client, fase 3) — a differenza della simulazione, dove raw_updates è il
      valore PRIMA del clip (stesso processo, ordine di codice diverso). Non è
      un mismatch per run_ids()/run_lira(): entrambi vogliono "la vista meno
      offuscata dal rumore DP disponibile al server", che per central DP è
      esattamente il valore clippato-non-rumorizzato — la stessa cosa,
      raggiunta per una via diversa (client-side invece che stessa riga di
      codice). Nessun round 0 viene esportato (l'Aggregator non vede mai i
      pesi di init casuale, generati da persistor/shareable_generator prima
      del round 1) — run_ids()/run_lira() gestiscono già round 0 assente
      (fallback a None, degradazione nota, non un crash).
    - _export_fl_results() fa pickle (non JSON: i GradientUpdate contengono
      torch.Tensor) dell'intera cronologia su disco dopo ogni round, stesso
      pattern overwrite-non-append di _export_results() (fase 4). Default:
      experiments/nvflare_fl_results.pkl, configurabile via
      `fl_results_export_path`.
    - Non ancora fatto: nessun no_dp bypass lato NVFLARE (l'Aggregator/
      Executor non hanno un flag equivalente a --no-dp della simulazione —
      dp_mode è sempre uno dei 3 valori, non c'è "disabilita tutto") — vedi
      scripts/run_nvflare_mia.py per come questo viene gestito (assume
      no_dp=False sempre quando chiama run_ids()/run_lira()).

Cosa NON fa ancora (fase 6+, vedi docs/NVFlareIntegration.md):
    - Nessuna analisi LiRA/Shadow/Yeom LIVE dentro aggregate() — per design
      (vedi sopra), resta un passo offline separato via
      scripts/run_nvflare_mia.py, eseguito dopo che il job NVFLARE finisce.
    - Non gestisce round con partecipazione parziale (drop-out) — stesso
      limite già segnalato nella review scalabilità/realismo di oggi per
      FedAvgAggregator (min_participants=tutti i client).

AGGIORNAMENTO (2026-08-07): due job NVFLARE reali completati end-to-end sulla
macchina dell'utente (10/10 round, metriche per-cliente distinte e non
degeneri — vedi experiments/nvflare_ids_audit_results_20260804_133100_58d089.json
e il suo .pkl gemello nvflare_fl_results_20260804_133100_58d089.pkl) confermano
empiricamente 3 dei 4 punti sotto elencati come "VERIFY". Il round_num resta
l'unico limite noto non risolvibile solo osservando run completati linearmente.

Punti CONFERMATI empiricamente dal job reale del 2026-08-04 (10/10 round
completati, metriche non degeneri — vedi
experiments/nvflare_fl_results_20260804_133100_58d089.pkl), ex-"VERIFY":
    - Formato di dxo.data in ingresso (dict[str state_dict_key, np.ndarray],
      identico a quanto emesso da chargeshield_executor.py): confermato — se
      il formato fosse stato diverso, accept() avrebbe sollevato KeyError o
      prodotto tensori corrotti invece di 10 round di aggregazione riuscita.
    - Come ottenere il node/cluster id di un client in accept() — dxo.meta
      ["cluster_id"] (impostato da chargeshield_executor.py): confermato
      accessibile e corretto in questo punto del ciclo di vita
      dell'Aggregator — il file di export mostra metriche per-client
      correttamente distinte tra "caltech"/"jpl"/"office1" round dopo round,
      non mescolate o mancanti.
    - Il formato di ritorno atteso da aggregate() — un DXO DataKind.WEIGHTS
      coi pesi aggregati completi: confermato che FullModelShareableGenerator
      lo interpreta correttamente come nuovo modello globale — i client hanno
      ricevuto e applicato con successo il modello di ogni round successivo
      (altrimenti l'intero job non avrebbe completato 10 round con loss
      sensata, si sarebbe bloccato o l'accuracy sarebbe rimasta a livello
      random-init).

Punto NON verificato (limite noto, non risolvibile osservando run completati
con successo — resta marcato inline con "VERIFY:"):
    - round_num: nessun contatore affidabile da fl_ctx trovato/provato in
      questa fase di scrittura — uso un contatore locale incrementato ad ogni
      aggregate(). I 2 job reali completati sono sempre andati linearmente
      round 1→10 senza interruzioni: in quello scenario un contatore locale e
      uno letto da fl_ctx (es. AppConstants.CURRENT_ROUND) coincidono, quindi
      il loro successo non prova nulla su questo punto. Fallirebbe
      silenziosamente (round_num disallineato rispetto allo stato server
      reale, senza eccezioni) SOLO in scenari mai testati finora: resume di
      un run interrotto, retry di un round fallito, o partecipazione parziale
      con round saltati per qualche client.

Punti aggiuntivi CONFERMATI per nvflare-sim ma NON per un deployment
Containerlab multi-container reale (2026-08-07, vedi i commenti inline su
_export_results()/_export_fl_results()): l'accesso in scrittura di
self._results_export_path/self._fl_results_export_path funziona nel run
single-process (`nvflare simulator`) che ha prodotto i due file reali sopra
citati, ma non è stato ancora esercitato in un deployment dove
l'Aggregator gira in un container server separato con il proprio bind mount
(containerlab/topology.clab.yml) — path assoluto e permessi in quello
scenario restano da verificare al primo deploy Containerlab end-to-end.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# BUG REALE trovato al primo run vero (2026-07-24, `make nvflare-sim-smoke`) —
# vedi il commento identico e più dettagliato in chargeshield_executor.py.
# In breve: `parents[4]` assumeva questo file fermo in nvflare/jobs/
# chargeshield_poc/app/custom/, ma `nvflare simulator` lo copia dentro il
# workspace (osservato: nvflare/sim_workspace/server/simulate_job/
# app_server/custom/) a una profondità diversa — risolveva _PROJECT_ROOT
# dentro sim_workspace/, causando "Auditor config not found:
# .../sim_workspace/config/auditor.yaml" in _ensure_components().
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
from nvflare.apis.fl_context import FLContext  # noqa: E402
from nvflare.apis.shareable import Shareable  # noqa: E402
from nvflare.app_common.abstract.aggregator import Aggregator  # noqa: E402

logger = logging.getLogger(__name__)

# FASE 7 (2026-08-31) — ML Plane realmente wired anche nel deployment NVFLARE:
# import a livello di modulo intenzionale (non lazy come ml.gradient_manager
# etc. in _ensure_components()) perché MLPlane/FLArtifactCollector/
# MLPlaneEvent non toccano torch/nvflare all'import — nessun rischio di
# introspezione prematura da parte di NVFLARE prima di START_RUN (lo stesso
# motivo per cui gli altri import restano lazy in _ensure_components()).
from ml.ml_plane import FLArtifactCollector, MLPlane  # noqa: E402
from ml.base_ml import MLPlaneEvent  # noqa: E402


class ChargeShieldAggregator(Aggregator):
    """
    Sostituisce InTimeAccumulateWeightedAggregator con FedAvgAggregator +
    PrivacyAuditor + ByzantineDetector + GradientManager reali (fase 2+3, vedi
    docstring del modulo).

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
                     run_ids() per la semantica. epsilon è usato ANCHE per
                     GradientManager quando dp_mode ne ha bisogno server-side
                     (dp-fedavg, central) — deve combaciare con l'epsilon del
                     client in config_fed_client.json.
        byzantine_tolerance / cosine_threshold / krum_threshold: passati a
                     ByzantineDetector — stessi default di run_ids() (0, 0.3, 3.5).
        dp_mode:     "dp-fedavg" (default) | "central" | "local" — deve
                     combaciare con dp_mode nel config del client. Vedi
                     docstring del modulo per cosa cambia server-side in
                     ciascun modo.
        delta:       parametro DP, passato a GradientManager insieme a
                     epsilon/max_grad_norm (fase 3).
        results_export_path: path (relativo alla root del progetto) del file
                     JSON dove viene scritta la cronologia IDS/Auditor
                     (fase 4) — sovrascritto per intero dopo ogni round.
                     Default: experiments/nvflare_ids_audit_results.json
                     (stessa dir, già in .gitignore, usata dalla simulazione).
        fl_results_export_path: path (relativo alla root del progetto) del
                     file PICKLE dove viene scritta, dopo ogni round, la
                     stessa struttura dati che run_fl_rounds() produce nella
                     simulazione (mean_loss, n_participants, updates,
                     raw_updates, raw_global_weights, global_weights) — fase
                     5, consumata offline da scripts/run_nvflare_mia.py per
                     eseguire LiRA/Shadow/Yeom SENZA reimplementarli qui.
                     Default: experiments/nvflare_fl_results.pkl.
    """

    def __init__(
        self,
        auditor_config_path: str = "config/auditor.yaml",
        # Fix 2026-07-22 (review indipendente fresh-pass): default stale dal
        # vecchio schema a 4 cluster fittizi. config_fed_server.json passa
        # sempre min_clients=3 esplicitamente (quindi la config attuale non era
        # affetta), ma questo default silenziosamente reintrodurrebbe il valore
        # sbagliato per qualunque altro caller/config futura che lo ometta —
        # allineato a 3, il numero di siti ACN-Data reali (caltech/jpl/office1).
        min_clients: int = 3,
        max_grad_norm: float = 1.0,
        epsilon: float | None = None,
        delta: float = 1.0e-5,
        explosion_threshold: float | None = None,
        byzantine_tolerance: int = 0,
        cosine_threshold: float = 0.3,
        krum_threshold: float = 3.5,
        dp_mode: str = "dp-fedavg",
        results_export_path: str = "experiments/nvflare_ids_audit_results.json",
        fl_results_export_path: str = "experiments/nvflare_fl_results.pkl",
        # FIX 2026-09-10 (bug reale trovato investigando perché il round 1 di
        # nvflare_ids_audit_results_*.json era byte-identico su TUTTI e 5 i
        # seed della campagna Step B, e anche sul run dp_mode="central"):
        # config_fed_client.json ha sempre avuto un campo "seed", ma veniva
        # usato SOLO lato client (chargeshield_executor.py: split train/holdout
        # + shuffle del DataLoader) — MAI per seedare l'inizializzazione dei
        # pesi del modello globale, che il persistor NVFLARE
        # (PTFileModelPersistor, componente successivo nella lista
        # "components" di config_fed_server.json) crea una sola volta,
        # deterministicamente uguale ad ogni submit_job, indipendentemente dal
        # seed dichiarato. A differenza della simulazione single-process
        # (scripts/run_experiments.py::main(), che chiama
        # random.seed/np.random.seed/torch.manual_seed(seed) PRIMA di
        # costruire il modello), qui non esisteva alcun punto che seedasse il
        # RNG globale di torch prima della costruzione del persistor. Fix:
        # nuovo parametro opt-in `seed` (default None, retro-compatibile con
        # qualunque job/test esistente che non lo passi — nessun cambio di
        # comportamento se omesso), seedato qui in __init__() PRIMA che
        # NVFLARE costruisca il componente successivo della lista (il
        # persistor, che istanzia l'Autoencoder) — l'ordine dei componenti in
        # config_fed_server.json (aggregator prima di persistor, già fissato
        # nel fix 2026-07-24 per un motivo di sys.path diverso) rende questo
        # sicuro: l'intero __init__ di ChargeShieldAggregator, incluso questo
        # seeding, completa prima che PTFileModelPersistor venga costruito.
        # config_fed_server.json aggiornato per passare lo STESSO valore di
        # "seed" già presente in config_fed_client.json (stesso principio
        # "DEVONO combaciare, nessuna validazione incrociata automatica" già
        # documentato per epsilon/delta/dp_mode). Non tocca in alcun modo le
        # conclusioni MIA già raccolte (LiRA/Yeom/Shadow leggono raw_updates
        # dal pickle, non l'inizializzazione del modello) — i 5 run dp-fedavg
        # e il run central già completati restano validi per quello che
        # misurano; il fix vale per le campagne NVFLARE future (es. la
        # variazione di epsilon o il DP mode "local" ancora da fare), che ora
        # avranno un round 1 genuinamente seed-dipendente come in simulazione.
        seed: int | None = None,
    ):
        super().__init__()
        self._auditor_config_path = str(_PROJECT_ROOT / auditor_config_path)
        self._seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            logger.info(
                f"ChargeShieldAggregator — seed={seed} applicato a "
                "random/numpy/torch PRIMA della costruzione del persistor "
                "(fix 2026-09-10: l'inizializzazione del modello globale ora "
                "dipende dal seed della campagna, come nella simulazione)."
            )
        else:
            logger.warning(
                "ChargeShieldAggregator — nessun seed passato: "
                "l'inizializzazione del modello globale NON è seedata "
                "(comportamento storico pre-2026-09-10) — round 1 identico "
                "tra run diversi che non passano questo parametro."
            )
        # Fix 2026-07-24 (bug reale trovato dall'utente sul primo
        # `make nvflare-sim-smoke -n 1`): min_clients=3 è corretto per il
        # deploy reale (3 siti ACN-Data), ma rende STRUTTURALMENTE impossibile
        # completare un'aggregazione con lo smoke test a 1 solo client (vedi
        # FedAvgAggregator.aggregate(): "partecipanti validi insufficienti:
        # 1 < 3" ogni round, aggregato sempre None) — non un crash, ma un
        # round vuoto per sempre, che impedisce allo smoke test di validare
        # qualunque cosa oltre al semplice round-trip DXO/accept(). Override
        # via env var (settata da `make nvflare-sim-smoke` a 1) permette allo
        # smoke test di completare un'aggregazione vera con un solo client,
        # senza toccare il default/config reale per il deploy a 3 siti.
        self._min_clients = int(
            __import__("os").environ.get("CHARGESHIELD_MIN_CLIENTS", min_clients)
        )
        self._max_grad_norm = max_grad_norm
        self._epsilon = epsilon
        self._delta = delta
        self._explosion_threshold = explosion_threshold
        self._byzantine_tolerance = byzantine_tolerance
        self._cosine_threshold = cosine_threshold
        self._krum_threshold = krum_threshold
        self._dp_mode = dp_mode
        # Fix 2026-07-24 (segnalato dall'utente dopo aver quasi perso i
        # risultati di uno smoke test riuscito a causa di un secondo run
        # avviato per sbaglio): a differenza di experiment_{timestamp}.json
        # nella simulazione (già unico per ogni run), questi due path erano
        # nomi FISSI — ogni nuovo run NVFLARE sovrascriveva silenziosamente
        # l'export del run precedente, riuscito o no. Ora un timestamp
        # catturato UNA volta qui (all'avvio di questo Aggregator, quindi una
        # volta per run) viene inserito nel nome file, stesso principio del
        # resto del progetto — nessun run futuro sovrascrive un run passato.
        # config_fed_server.json resta invariato (i due valori di default
        # sopra restano "puliti"); il timestamp è aggiunto qui, non lì.
        #
        # Fix 2026-07-24 (review indipendente, stesso giorno): il timestamp
        # aveva risoluzione al secondo — due run avviati nello stesso secondo
        # (es. due processi paralleli, possibile in un futuro sweep NVFLARE
        # parallelizzato) avrebbero prodotto lo stesso nome file e si sarebbero
        # sovrascritti a vicenda esattamente come nel bug originale. Aggiunto
        # un suffisso random (non basato su un controllo "esiste già" — quel
        # pattern ha la stessa race condition se due processi lo eseguono nello
        # stesso istante) per rendere la collisione trascurabile anche fra run
        # concorrenti, mantenendo il timestamp leggibile come prefisso.
        from datetime import datetime as _datetime
        from uuid import uuid4 as _uuid4

        _run_ts = _datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + _uuid4().hex[:6]

        def _with_run_timestamp(path_str: str, ts: str) -> Path:
            p = _PROJECT_ROOT / path_str
            return p.with_name(f"{p.stem}_{ts}{p.suffix}")

        self._results_export_path = _with_run_timestamp(results_export_path, _run_ts)
        self._fl_results_export_path = _with_run_timestamp(fl_results_export_path, _run_ts)
        # Cronologia IDS/Auditor per-round (fase 4) — {round_num: {...}},
        # scritta per intero su self._results_export_path dopo ogni round.
        self._audit_history: dict[int, dict[str, Any]] = {}
        # Cronologia "fl_results"-compatibile per-round (fase 5) — stesso
        # schema di run_fl_rounds(), scritta per intero (pickle) su
        # self._fl_results_export_path dopo ogni round.
        self._fl_results_history: dict[int, dict[str, Any]] = {}

        # Istanziati lazy in _ensure_components() — evita di importare
        # torch/ml/auditor/ids al momento della definizione della classe
        # (che NVFLARE può introspezionare prima di START_RUN).
        self._fedavg = None
        self._auditor = None
        self._ids = None
        self._gm = None  # GradientManager (fase 3) — serve per dp_mode="dp-fedavg"/"central"
        # FASE 7 (2026-08-31, richiesta esplicita dell'autore dopo il feedback
        # su "il ML Plane non è realmente usato nel deployment NVFLARE" —
        # confermato leggendo il codice: prima di questa fase, questo file non
        # importava né istanziava MLPlane/FLArtifactCollector, e accept()/
        # aggregate() costruivano _fl_results_history direttamente da variabili
        # Python locali (received_updates, updates_for_fedavg) — esattamente lo
        # stesso pattern "morto" che src/ml/ml_plane.py aveva già corretto nella
        # simulazione (scripts/run_experiments.py::run_fl_rounds()) il
        # 2026-07-22, ma mai riportato qui). Istanziati in _ensure_components(),
        # stesso principio lazy-import degli altri componenti sotto.
        self._mlplane: MLPlane | None = None
        self._collector: FLArtifactCollector | None = None
        # FASE 8 (2026-08-31) — vedi _ensure_components()/PrivacyAuditorSubscriber
        # in src/auditor/privacy_auditor.py.
        self._auditor_subscriber: Any = None
        # BUG REALE trovato al primo run vero (2026-07-24, `make nvflare-sim-smoke`):
        # _ensure_components() usava "if self._fedavg is not None: return" come
        # guardia di "già inizializzato". Al round 0, PrivacyAuditor(...) ha
        # sollevato FileNotFoundError (per il bug _PROJECT_ROOT sopra, ora
        # corretto) A META' di _ensure_components() — DOPO che self._fedavg era
        # già stato assegnato, ma PRIMA di self._gm. La chiamata a
        # _ensure_components() del round SUCCESSIVO ha quindi visto
        # self._fedavg non-None, concluso "già pronto" e saltato la
        # re-inizializzazione — lasciando self._gm permanentemente None anche
        # dopo che la causa originale (path sbagliato) sarebbe stata risolta.
        # Sintomo osservato: "AttributeError: 'NoneType' object has no
        # attribute 'privatize'" in aggregate() al round 1 — un crash che
        # sembrava un bug diverso, ma era solo il secondo effetto dello stesso
        # fallimento parziale. Fix: flag esplicito impostato SOLO a fine
        # inizializzazione riuscita, cosi' un fallimento parziale permette un
        # retry completo al prossimo round invece di un "successo" fittizio.
        self._components_ready = False

        self._weight_keys: list[str] | None = None
        self._round_updates: list[Any] = []   # GradientUpdate raccolti questo round
        # LIMITE NOTO (non un VERIFY generico, vedi docstring del modulo):
        # contatore locale, mai confrontato con AppConstants.CURRENT_ROUND da
        # fl_ctx. I 2 job NVFLARE reali completati (10/10 round, sempre
        # lineari 1→10) non possono confermare né smentire questo punto —
        # fallirebbe silenziosamente solo in scenari di resume/retry/round-skip
        # mai ancora testati.
        self._round_num = 0
        # Modello globale distribuito ai client all'INIZIO del round corrente
        # (= output dell'aggregate() precedente) — usato sia come riferimento
        # per il delta peer-relative dell'IDS, sia come reference_weights per
        # GradientManager.privatize() lato server in dp_mode="dp-fedavg".
        # Rinominato da _prev_raw_global (fase 2) perché con la DP cablata
        # (fase 3) non è più necessariamente "raw" per ogni dp_mode.
        self._prev_global_weights: list[Any] | None = None
        # NOTA (Fix 2026-07-22, superata dalla FASE 8 del 2026-08-31): questo
        # progetto teneva qui un secondo campo, _prev_raw_global_weights, come
        # baseline SEPARATA (pulita, pre-DP) per il delta peer-relative
        # dell'IDS — necessaria perché _prev_global_weights sopra è POST-DP
        # dal round 2 in poi, e usarlo come baseline gonfierebbe i delta con
        # il rumore del round precedente (falsi GRADIENT_EXPLOSION/Krum).
        # Da quando PrivacyAuditor è un vero subscriber ML Plane
        # (PrivacyAuditorSubscriber, src/auditor/privacy_auditor.py), questa
        # stessa baseline è mantenuta INTERNAMENTE al subscriber
        # (self._auditor_subscriber, popolata SOLO dalla vista raw — mai
        # dalla privatizzata, stessa semantica di prima) — il campo qui non
        # serve più ed è stato rimosso per evitare due fonti di verità per
        # lo stesso concetto.

    # ── Lazy init ────────────────────────────────────────────────────────────

    def _ensure_components(self) -> None:
        # Fix 2026-07-24 (bug reale, vedi commento su self._components_ready
        # nell'__init__): guardia sul flag esplicito, non su "self._fedavg is
        # not None" — quella permetteva a un fallimento parziale (un'eccezione
        # a metà inizializzazione) di essere scambiato per "già pronto" al
        # round successivo, lasciando self._gm/_auditor/_ids permanentemente
        # None. Con questo fix, un'eccezione qui NON imposta _components_ready,
        # quindi il prossimo round ritenta l'inizializzazione completa da zero.
        if self._components_ready:
            return
        from ml.fedavg_aggregator import FedAvgAggregator
        from ml.gradient_manager import GradientManager
        from auditor.privacy_auditor import PrivacyAuditor
        from auditor.privacy_auditor_subscriber import PrivacyAuditorSubscriber
        from ids.charging_ids import ByzantineDetector

        self._fedavg = FedAvgAggregator({"min_participants": self._min_clients})
        self._auditor = PrivacyAuditor(
            config_path=self._auditor_config_path,
            epsilon=self._epsilon,
            explosion_threshold=self._explosion_threshold,
        )
        self._ids = ByzantineDetector(
            config_path=self._auditor_config_path,
            byzantine_tolerance=self._byzantine_tolerance,
            cosine_threshold=self._cosine_threshold,
            krum_threshold=self._krum_threshold,
        )
        # GradientManager: serve solo per dp_mode="dp-fedavg" (privatize per-client
        # server-side) e "central" (privatize_aggregate sull'aggregato) — istanziato
        # comunque per semplicità, inutilizzato in dp_mode="local" (tutto client-side).
        self._gm = GradientManager({
            "epsilon": self._epsilon if self._epsilon is not None else 1.0,
            "delta": self._delta,
            "max_grad_norm": self._max_grad_norm,
        })

        # FASE 7 (2026-08-31) — ML Plane reale: stesso hub/collector usati in
        # run_fl_rounds() (scripts/run_experiments.py), stessa identità di
        # classi (src/ml/ml_plane.py), non una re-implementazione. wire()
        # registra il collector come subscriber di GradientManager e
        # FedAvgAggregator (entrambi già emettono emit_event() reali — vedi
        # gradient_manager.py/fedavg_aggregator.py); l'Aggregator NVFLARE
        # stesso NON eredita AbstractMLModel/subscribe(), quindi per il
        # confine "client update arrivato in accept(), prima di FedAvg" (il
        # punto esatto in cui il paper QRS 2026 colloca il Privacy Auditor —
        # vedi docstring di modulo) emette direttamente su self._mlplane
        # invece di passare per wire() — vedi accept() sotto.
        self._mlplane = MLPlane()
        self._collector = FLArtifactCollector()
        self._mlplane.subscribe(self._collector)
        self._mlplane.wire(self._gm, self._fedavg)

        # FASE 8 (2026-08-31) — Privacy Auditor come vero subscriber ML Plane
        # (non più invocato imperativamente da _run_ids_analysis(), vedi
        # PrivacyAuditorSubscriber in src/auditor/privacy_auditor.py per il
        # design completo). Sottoscritto allo STESSO self._mlplane di
        # self._collector — riceve quindi anche gli eventi "gradient_upload"
        # emessi direttamente da accept() (non via wire(), vedi commento lì),
        # oltre a quelli di GradientManager/FedAvgAggregator via wire().
        self._auditor_subscriber = PrivacyAuditorSubscriber(
            self._auditor, self._collector, self._max_grad_norm,
        )
        self._mlplane.subscribe(self._auditor_subscriber)
        # Nessuna baseline round-0 disponibile nel vero NVFLARE (stesso KNOWN
        # GAP già documentato sopra per self._prev_global_weights al round 1
        # — l'Aggregator non vede mai i pesi di init pre-round-1) — resta
        # None, comportamento invariato rispetto a _run_ids_analysis() prima
        # di questa fase.

        self._components_ready = True  # solo qui, dopo che TUTTO sopra è riuscito
        logger.info(
            f"ChargeShieldAggregator inizializzato — FedAvgAggregator + "
            f"PrivacyAuditor + ByzantineDetector + GradientManager + ML Plane "
            f"(fase 7, dp_mode={self._dp_mode})"
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

        # CONFERMATO empiricamente dal job reale del 2026-08-04 (10/10 round
        # completati, metriche non degeneri — vedi
        # experiments/nvflare_fl_results_20260804_133100_58d089.pkl): dxo.data
        # è dict[str state_dict_key, np.ndarray], stesso formato emesso da
        # chargeshield_executor.py.
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

        # CONFERMATO empiricamente dal job reale del 2026-08-04: cluster_id
        # preso da dxo.meta (impostato da chargeshield_executor.py), non da
        # fl_ctx/peer info — accessibile in questo punto del ciclo di vita
        # dell'Aggregator e sufficiente a identificare correttamente il client
        # mittente (l'export finale mostra metriche per-cluster distinte e
        # coerenti per caltech/jpl/office1 round dopo round, mai mescolate).
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

        # FASE 7 (2026-08-31) — ML Plane reale: questo È il punto in cui il
        # paper QRS 2026 colloca il Privacy Auditor ("client updates are
        # temporarily available in server memory before the execution of
        # FedAvg") — il momento in cui l'update di UN client attraversa
        # davvero il confine verso il server, nel vero deployment NVFLARE
        # (a differenza della simulazione, dove questo confine è emulato da
        # train_local() nello stesso processo). Il livello Purdue dipende da
        # dp_mode, stessa semantica già usata da _raw_updates_for_export più
        # sotto (mai una re-invenzione):
        #   - "dp-fedavg": il client NON applica DP — questo è genuinamente
        #     RAW (purdue_level=1); il server lo privatizza qui in
        #     aggregate() (gm.privatize(), che emette il purdue_level=2).
        #   - "central": il client ha già CLIPPATO (non rumorizzato) — trattato
        #     come "raw" ai fini IDS/export (stessa scelta già presente nel
        #     codice esistente, vedi commento su _raw_updates_for_export in
        #     aggregate()), quindi purdue_level=1 anche qui.
        #   - "local": il client ha già clippato E rumorizzato — il server
        #     non vede MAI nulla di meno rumoroso di questo. Emettere questo
        #     come purdue_level=1 ("raw") sarebbe scorretto: non esiste, lato
        #     server, nulla di più raw. Emesso invece come purdue_level=2
        #     (privatizzato) — a differenza della simulazione (dove
        #     train_local() emette SEMPRE purdue_level=1 anche sotto local DP,
        #     e solo _store_raw lo nasconde a valle, vedi run_fl_rounds()),
        #     qui il confine è strutturalmente rispettato: il ML Plane del
        #     server non raccoglie mai un evento "raw" per local DP, non solo
        #     non lo esporta. Differenza voluta, non un'incoerenza col
        #     comportamento della simulazione — da documentare esplicitamente
        #     nel paper come punto di forza del deployment reale.
        purdue_level = 2 if self._dp_mode == "local" else 1
        self._mlplane.on_ml_event(MLPlaneEvent(
            event_type="gradient_upload",
            purdue_level=purdue_level,
            payload=update,
            round_num=self._round_num + 1,
            metadata={"dp_mode": self._dp_mode, "nvflare_boundary": "accept"},
        ))

        logger.debug(f"ChargeShieldAggregator.accept: raccolto update da [{node_id}]")
        return True

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """Chiamato da ScatterAndGather dopo che tutti i client (o il timeout)
        hanno risposto per il round corrente. Esegue FedAvg + IDS/Auditor,
        poi restituisce il nuovo modello globale come DXO/Shareable."""
        self._ensure_components()
        import numpy as np

        # LIMITE NOTO (vedi commento su self._round_num nell'__init__ e
        # docstring del modulo): non testato in scenari di resume/retry/
        # round-skip.
        self._round_num += 1
        received_updates = self._round_updates
        self._round_updates = []

        if not received_updates:
            logger.warning(f"Round {self._round_num}: nessun update ricevuto — aggregazione saltata")
            return DXO(data_kind=DataKind.WEIGHTS, data={}).to_shareable()

        # ── DP fase 3: cosa arriva in received_updates dipende da dp_mode ──────
        # - "dp-fedavg": RAW (il client non ha applicato DP) — l'IDS analizza
        #   questi stessi valori raw (mirroring run_ids()), poi li privatizziamo
        #   UNO PER UNO qui prima di passarli a FedAvg — placement per-client
        #   non-standard, non descritto letteralmente nell'Algoritmo 1 di
        #   McMahan et al. 2018 (quel paper corrisponde invece al mode
        #   "central", vedi sotto).
        # - "central": già clippati (non rumorizzati) dal client — l'IDS li
        #   analizza così, FedAvg li combina, POI aggiungiamo un solo rumore
        #   all'aggregato (privatize_aggregate(), sotto).
        # - "local": già clippati+rumorizzati dal client — l'IDS analizza dati
        #   già rumorizzati (degradazione attesa, stessa nota di run_ids()).
        # KNOWN GAP (review indipendente fase 3-5, non risolto): al round 1,
        # self._prev_global_weights è ancora None (viene assegnato solo alla
        # fine di aggregate(), la prima volta dopo che il round 1 è già stato
        # aggregato) — reference_weights=None fa scattare in
        # GradientManager._clip_weights() il fallback storico (clip sul
        # vettore ASSOLUTO, non sul delta), diversamente da ogni round
        # successivo (dove self._prev_global_weights è il vero modello globale
        # precedente) e diversamente dalla simulazione (dove pre_round_weights
        # in run_fl_rounds() è SEMPRE concreto, anche al round 1 — è il modello
        # random-init di ogni trainer, disponibile perché client e server sono
        # lo stesso processo). Nel vero NVFLARE l'Aggregator non ha modo di
        # vedere il modello di inizializzazione che persistor/shareable_
        # generator inviano ai client PRIMA del round 1 — un fix corretto
        # richiederebbe che il client includa i propri pesi pre-round nel DXO
        # (plumbing aggiuntivo, non fatto qui: tentare un fix speculativo senza
        # poter eseguire nulla in questo sandbox rischierebbe di introdurre
        # un'assunzione sbagliata invece di una nota onesta). Effetto pratico:
        # SOLO il clipping DP-FedAvg del round 1 usa la semantica "assoluta"
        # invece che "delta" — dal round 2 in poi il comportamento è corretto.
        if self._dp_mode == "dp-fedavg":
            for u in received_updates:
                self._gm.privatize(
                    u, weight_keys=self._weight_keys, reference_weights=self._prev_global_weights,
                )
            # gm.privatize() sopra ha già emesso un evento purdue_level=2 per
            # ciascun update — non serve più raccogliere il risultato in una
            # lista locale.

        # FASE 7 (2026-08-31) — ML Plane reale: updates_for_fedavg letto dal
        # collector, non da una lista costruita a mano — stesso principio di
        # `_collected_updates = collector.privatized_updates(round_num)` in
        # run_fl_rounds(). Semantica per dp_mode (invariata rispetto a prima,
        # ora genuinamente sourced dagli eventi ML Plane):
        #   - "dp-fedavg": gm.privatize() sopra ha appena emesso purdue_level=2
        #     per ogni update di questo round — restituiti direttamente.
        #   - "central": nessun evento purdue_level=2 per questo round (il
        #     server non ri-privatizza per-client) → privatized_updates()
        #     ricade su raw_updates() (purdue_level=1, emesso in accept() —
        #     gli update già clippati dal client), identico a
        #     `received_updates` di prima.
        #   - "local": accept() ha già emesso purdue_level=2 direttamente
        #     (vedi commento lì) → restituiti senza fallback, identico a
        #     `received_updates` di prima.
        updates_for_fedavg = self._collector.privatized_updates(self._round_num)

        for u in updates_for_fedavg:
            self._fedavg.collect(u)
        aggregated = self._fedavg.aggregate(self._round_num)

        # ── IDS/Auditor: replica semplificata di run_ids() per questo round ──
        # Usa SEMPRE received_updates (la vista più "raw" disponibile in questo
        # dp_mode), non updates_for_fedavg — stesso principio di run_ids() che
        # preferisce raw_updates quando esistono.
        self._run_ids_analysis(received_updates)

        # ── Fase 5: raw-update extraction per LiRA/Shadow (vedi docstring modulo) ──
        # FASE 7 (2026-08-31): letto dal collector (self._collector.raw_updates()),
        # non da received_updates direttamente — stessa semantica di prima
        # (None sotto "local": accept() non emette mai purdue_level=1 in quel
        # dp_mode, quindi raw_updates() restituirebbe [] — normalizzato
        # esplicitamente a None qui per preservare il contratto esistente di
        # _fl_results_history["raw_updates"], letto da scripts/run_nvflare_mia.py
        # e coerente con _store_raw in run_fl_rounds()), ora genuinamente
        # sourced dagli eventi ML Plane invece che dalla lista locale.
        _raw_updates_for_export = (
            self._collector.raw_updates(self._round_num) if self._dp_mode != "local" else None
        )
        _raw_global_weights_for_export = (
            self._weighted_average_weights(_raw_updates_for_export)
            if _raw_updates_for_export else None
        )

        if aggregated is None or not aggregated.global_weights:
            logger.error(
                f"Round {self._round_num}: FedAvgAggregator non ha prodotto un "
                "aggregato (partecipanti insufficienti?) — restituisco Shareable vuoto"
            )
            return DXO(data_kind=DataKind.WEIGHTS, data={}).to_shareable()

        # Central DP (fase 3): un solo rumore Gaussiano sull'aggregato pulito,
        # DOPO FedAvg — mirroring il blocco equivalente in run_fl_rounds().
        if self._dp_mode == "central":
            aggregated.global_weights = self._gm.privatize_aggregate(
                aggregated.global_weights,
                weight_keys=self._weight_keys,
                n_participants=aggregated.n_participants or self._min_clients,
                # FASE 8 (2026-08-31, fix sensibilità pesata) — vedi
                # GradientManager.privatize_aggregate() per il perché.
                participant_n_samples=list(
                    aggregated.metadata.get("participant_n_samples", {}).values()
                ),
            )
            # FASE 7 (2026-08-31) — ML Plane reale: self._fedavg.aggregate()
            # sopra ha già emesso un evento "aggregation" con l'aggregato
            # PULITO (pre-rumore central). privatize_aggregate() modifica
            # aggregated.global_weights in-place — senza questa ri-emissione,
            # self._collector.aggregation(round_num) restituirebbe l'aggregato
            # sbagliato (senza rumore), mentre è quello rumorizzato che viene
            # davvero distribuito ai client — stessa identica ri-emissione già
            # presente in run_fl_rounds() (scripts/run_experiments.py) per lo
            # stesso motivo.
            self._fedavg.emit_event(MLPlaneEvent(
                event_type="aggregation",
                purdue_level=3,
                payload=aggregated,
                round_num=self._round_num,
            ))

        # Aggiorna il riferimento per il clip server-side di dp-fedavg al
        # prossimo round — è il modello che verrà distribuito a tutti i client
        # come punto di partenza del round successivo (POST eventuale rumore
        # central-DP sopra — corretto qui, perché è esattamente "il modello che
        # il client riceverà").
        self._prev_global_weights = aggregated.global_weights
        # NOTA (FASE 8, 2026-08-31): la baseline separata e pulita (pre-DP)
        # per l'IDS/Auditor non è più mantenuta qui — self._auditor_subscriber
        # l'ha già aggiornata da sé, reagendo all'evento "aggregation" appena
        # emesso sopra (vedi PrivacyAuditorSubscriber._handle_round_complete()).

        # ── Fase 5 (continua): entry fl_results-compatibile per questo round ──
        # Stesso schema esatto di run_fl_rounds() (vedi docstring modulo) —
        # global_weights qui è già POST eventuale rumore central-DP sopra,
        # identico a ciò che viene ridistribuito ai client.
        self._fl_results_history[self._round_num] = {
            "mean_loss": aggregated.mean_loss,
            "n_participants": aggregated.n_participants,
            "updates": updates_for_fedavg,
            "raw_updates": _raw_updates_for_export,
            "raw_global_weights": _raw_global_weights_for_export,
            "global_weights": aggregated.global_weights,
        }
        self._export_fl_results()

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
        FASE 8 (2026-08-31): non calcola più delta/norme/audit a mano — li
        legge da self._auditor_subscriber (src/auditor/privacy_auditor.py::
        PrivacyAuditorSubscriber), un vero subscriber ML Plane che ha già
        reagito all'evento "aggregation" di questo round (emesso da
        self._fedavg.aggregate(), chiamato PRIMA di questo metodo in
        aggregate() sopra — dispatch sincrono, quindi i report sono già
        pronti quando arriviamo qui). Stessa formula esatta di prima (mediana
        → max_grad_norm, fix Sprint 9 GRADIENT_EXPLOSION) — vedi
        PrivacyAuditorSubscriber per il dettaglio completo, non duplicato qui.

        Il parametro `updates` (ancora passato dal chiamante per compatibilità
        di firma) non è più usato per il calcolo: il subscriber legge
        autonomamente da self._collector, che ha la stessa identica vista
        "raw preferito, privatizzato come fallback" che `updates` rappresentava.

        Fase 4 (2026-07-22, sera): costruisce una entry strutturata in
        self._audit_history[self._round_num] nello stesso formato di
        ids_results in run_ids() (vedi quella funzione), e la esporta su
        JSON via _export_results() — invariato rispetto a prima.
        """
        reports = self._auditor_subscriber.reports_for_round(self._round_num)
        gradients = self._auditor_subscriber.gradients_for_round(self._round_num)

        if not reports:
            self._audit_history[self._round_num] = {
                "alerts": [], "byzantine_detected": False, "low_similarity_nodes": [],
                "per_client_audit": {},
            }
            self._export_results()
            return

        analysis = self._ids.analyze_round(self._round_num, reports, gradients)
        if getattr(analysis, "byzantine_nodes", None):
            logger.warning(
                f"Round {self._round_num}: ByzantineDetector ha rilevato nodi Byzantine: "
                f"{analysis.byzantine_nodes}"
            )
        else:
            logger.debug(f"Round {self._round_num}: nessun nodo Byzantine rilevato")

        # ── Fase 4: entry strutturata, stesso formato di ids_results in run_ids() ──
        self._audit_history[self._round_num] = {
            "alerts": [
                {
                    "node_id": a.node_id,
                    "severity": a.severity,
                    "reasons": a.reasons,
                    "recommended_action": a.recommended_action,
                }
                for a in (analysis.alerts if analysis else [])
            ],
            "byzantine_detected": bool(analysis.byzantine_nodes) if analysis else False,
            "low_similarity_nodes": analysis.low_similarity_nodes if analysis else [],
            "per_client_audit": {
                node_id: {
                    "privacy_score": report.privacy_score,
                    "epsilon": report.epsilon,
                    "threats_detected": report.threats_detected,
                }
                for node_id, report in reports.items()
            },
        }
        self._export_results()

    def _export_results(self) -> None:
        """
        Scrive self._audit_history per intero su self._results_export_path
        (overwrite, non append) — chiamata alla fine di ogni round, cosi' il
        file riflette sempre lo stato piu' recente anche se il job si ferma
        a meta'.

        CONFERMATO per il caso nvflare-sim (2026-08-04): l'accesso in
        scrittura a _PROJECT_ROOT/experiments/ funziona — i due job reali
        completati (10/10 round) hanno prodotto per davvero
        experiments/nvflare_ids_audit_results_20260804_133100_58d089.json
        via questo stesso metodo. Quel run è però un singolo processo Python
        (client ed Aggregator nello stesso filesystem/working directory, via
        `nvflare simulator` — vedi Makefile `nvflare-sim`), non un vero
        deployment multi-container. NON ANCORA CONFERMATO per un deployment
        Containerlab reale (server in un container separato con il proprio
        bind mount — vedi containerlab/topology.clab.yml), dove path assoluto
        e permessi di scrittura del container server non sono stati ancora
        esercitati end-to-end.
        """
        try:
            self._results_export_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "config": {
                    "dp_mode": self._dp_mode,
                    "epsilon": self._epsilon,
                    "delta": self._delta,
                    "max_grad_norm": self._max_grad_norm,
                    "min_clients": self._min_clients,
                    "krum_threshold": self._krum_threshold,
                    "cosine_threshold": self._cosine_threshold,
                },
                "per_round": {str(r): v for r, v in sorted(self._audit_history.items())},
            }
            with open(self._results_export_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
        except OSError as exc:
            # Non deve mai far fallire il round FL per un errore di I/O sui
            # risultati — logga e continua (stesso principio difensivo di
            # accept() sopra: un errore qui non deve bloccare l'addestramento).
            logger.error(f"ChargeShieldAggregator._export_results: scrittura fallita: {exc}")

    # ── Fase 5: raw-update extraction per LiRA/Shadow ───────────────────────

    @staticmethod
    def _weighted_average_weights(updates: list[Any]) -> list[Any] | None:
        """
        Media pesata (per n_samples) dei pesi di una lista di GradientUpdate —
        STESSA formula usata per raw_global_weights in run_fl_rounds()
        (scripts/run_experiments.py, righe vicino a "Calcola raw_global_weights").
        Restituisce None se updates è vuota o il primo update non ha pesi.
        """
        import torch

        if not updates or not updates[0].weights:
            return None
        n_w = len(updates[0].weights)
        total = sum(u.n_samples for u in updates) or len(updates)
        averaged = []
        for i in range(n_w):
            wavg = sum(
                (u.weights[i] if isinstance(u.weights[i], torch.Tensor)
                 else torch.tensor(float(u.weights[i])))
                * (u.n_samples / total)
                for u in updates
            )
            averaged.append(wavg)
        return averaged

    def _export_fl_results(self) -> None:
        """
        Scrive self._fl_results_history per intero (pickle) su
        self._fl_results_export_path dopo ogni round — stesso pattern
        overwrite-non-append di _export_results() (fase 4), stessa
        motivazione difensiva (mai far fallire il round FL per un errore
        di I/O). Pickle invece di JSON: gli elementi di "updates"/
        "raw_updates" sono oggetti ml.base_ml.GradientUpdate contenenti
        torch.Tensor, non serializzabili in JSON senza perdita di fedeltà —
        e la lettura (scripts/run_nvflare_mia.py) li passa direttamente,
        invariati, a run_lira()/run_ids() che si aspettano esattamente
        questo tipo.

        Stessa nota di _export_results() sopra: CONFERMATO per nvflare-sim
        (i .pkl reali in experiments/nvflare_fl_results_*.pkl sono prodotti
        da questo stesso metodo), NON ANCORA CONFERMATO per un deployment
        Containerlab multi-container reale.
        """
        import pickle

        try:
            self._fl_results_export_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "meta": {
                    "dp_mode": self._dp_mode,
                    "epsilon": self._epsilon,
                    "delta": self._delta,
                    "max_grad_norm": self._max_grad_norm,
                },
                "rounds": dict(self._fl_results_history),
            }
            with open(self._fl_results_export_path, "wb") as f:
                pickle.dump(payload, f)
        except OSError as exc:
            logger.error(f"ChargeShieldAggregator._export_fl_results: scrittura fallita: {exc}")
