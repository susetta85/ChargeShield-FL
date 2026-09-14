# src/adapters/chargeplace_scotland_adapter.py
"""
ChargePlace Scotland Dataset Adapter
=====================================
Traduce il dataset pubblico ChargePlace Scotland (rete di ricarica EV
gestita da Transport Scotland / Scottish Government) dal formato Excel
grezzo al formato standard del framework ChargeShield-FL — stesso
contratto (AbstractDataset, stessi FEATURE_NAMES) usato da
src/adapters/acn_dataset.py, per permettere di riusare l'intera pipeline
FL/MIA senza modifiche a valle.

Dataset source: portale ChargePlace Scotland / richiesta dati a Transport
Scotland (vedi docs/TestRoadmap_DSN2027.md, roadmap #6 / task #37).
Formato input: file .xlsx mensili in Sessions_from_CPS/ + file di
metadati per CPID (charge-point ID) in CPID_information/.

Verificato manualmente sui file reali (2026-09-09): 18 file mensili,
3.120.526 sessioni totali (NON ~3.9M come riportato in
docs/TestRoadmap_DSN2027.md — numero da correggere lì, vedi Sprint-log).

Differenze strutturali rispetto ad ACN-Data, rilevanti per chi userà
questo adapter a valle:

- Nessun concetto di "sito fisico" nel file sessioni: il raggruppamento
  usato come site_id è il local_authority (32 council area scozzesi),
  preso da CPID_and_local_authority.xlsx e unito per CPID. Analogo
  concettualmente al siteID di ACN-Data (Caltech/JPL/Office1), ma con
  granularità molto più fine (32 invece di 3) — quali/quante council
  area usare come client FL è una decisione a valle, non di questo
  modulo. Copertura join verificata: 99.93% delle sessioni ha un CPID
  presente nella mappa metadati (162783/162896 su OCT-23.xlsx); le
  sessioni senza match ottengono site_id="" (charge point dismesso o
  aggiunto dopo lo snapshot dei metadati).
- Nessun user_id: ChargePlace Scotland non traccia l'utente, solo il
  punto di ricarica (CPID). user_id è sempre None. LIMITE NOTO: lo
  split train/holdout entity-aware (README task #27/#38) non è
  replicabile su questo dataset a livello utente, solo a livello CPID
  (già coperto da node_id) — da documentare esplicitamente se/quando
  si useranno questi dati per un esperimento di quel tipo.
- Nessun session_id nativo: sintetizzato come hash deterministico di
  (CPID, start_time) — vedi _synthesize_session_id().
- 'Duration' è salvata da Excel come cella "ora del giorno"
  (datetime.time): vedi il warning in _parse_duration_to_timedelta().
- Non esiste un doneChargingTime separato: done_charging_time ==
  end_time sempre (a differenza di ACN-Data non c'è modo di distinguere
  "fine ricarica" da "disconnessione").
- kwh_requested e minutes_available non esistono in questo dataset
  (nessun equivalente di userInputs di ACN-Data) → sempre 0.0 / 0,
  MAI usati come proxy di richiesta utente in analisi a valle.

Questo adapter NON conosce FL, protocolli, o il Privacy Auditor.
"""

import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

from core.base_dataset import AbstractDataset

# ChargePlace Scotland opera esclusivamente in Scozia.
_TIMEZONE = "Europe/London"
_LOCAL_TZ = ZoneInfo(_TIMEZONE)
_UTC = ZoneInfo("UTC")

# Colonne attese nei file mensili Sessions_from_CPS/*.xlsx.
_REQUIRED_COLUMNS = {"CPID", "Consumed(kWh)", "Duration", "Start", "Time"}


def _parse_duration_to_timedelta(value: Any) -> timedelta:
    """
    Converte il valore della colonna 'Duration' in timedelta.

    pandas/openpyxl leggono le celle Excel con durata < 24h come
    datetime.time. Per le sessioni >= 24h (formato Excel elapsed-time
    "[h]:mm:ss") il valore arriva invece come STRINGA nel formato nativo
    di str(datetime.timedelta) — es. "1 day, 8:37:17" o "2 days, 17:15:00"
    — MAI wrappato modulo 24h: verificato empiricamente (2026-09-09) che
    non c'è alcun troncamento, contrariamente al sospetto iniziale.
    Trovato con i dati reali: 993/162896 sessioni in OCT-23.xlsx hanno
    Duration in questo formato stringa (non un caso raro/limite — quasi
    l'1% del file). Un semplice str(value).split(":") su queste stringhe
    fallisce (int("1 day, 8") solleva ValueError) — bug reale scoperto
    scrivendo i test di questo adapter, corretto qui usando pd.Timedelta,
    che interpreta nativamente sia "HH:MM:SS" sia "N day(s), H:MM:SS".

    Le 7 sessioni con Duration mancante (NaN, verificato) vengono scartate
    a monte in _parse_file() come record malformati, non arrivano qui.
    """
    if isinstance(value, timedelta):
        return value
    if hasattr(value, "hour"):  # datetime.time (< 24h, il caso comune)
        return timedelta(hours=value.hour, minutes=value.minute, seconds=value.second)
    # Fallback robusto: gestisce sia "HH:MM:SS" sia "N day(s), H:MM:SS".
    return pd.Timedelta(str(value)).to_pytimedelta()


def _compute_max_power_kw(kwh: float, duration: timedelta) -> float:
    """Stima la potenza media in kW: energia erogata / ore di ricarica. 0.0 se durata <= 0."""
    duration_hours = duration.total_seconds() / 3600.0
    if duration_hours <= 0:
        return 0.0
    return round(kwh / duration_hours, 3)


def _synthesize_session_id(cpid: str, start: datetime) -> str:
    """
    ChargePlace Scotland non fornisce un session_id nativo (a differenza di
    ACN-Data). Lo sintetizziamo come hash deterministico di (CPID,
    start_time) — stabile tra run diverse dello stesso file, univoco quanto
    la granularità al secondo del campo 'Time' (due sessioni sullo stesso
    CPID che iniziano nello stesso secondo collidono: non osservato nei
    dati campione ma teoricamente possibile).
    """
    raw = f"{cpid}|{start.isoformat()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class ChargePlaceScotlandDataset(AbstractDataset):
    """
    Adapter per il dataset ChargePlace Scotland (file Excel mensili).

    Uso tipico:
        ds = ChargePlaceScotlandDataset()
        ds.load_with_metadata(
            session_paths=["datasets/alt/chargeplace_scotland/Sessions_from_CPS/OCT-23.xlsx"],
            metadata_dir="datasets/alt/chargeplace_scotland/CPID_information",
        )
        sample = ds.get_sample(0)
        print(sample["site_id"])  # es. "City of Edinburgh"

    load()/load_multiple() (contratto AbstractDataset) caricano le
    sessioni SENZA i metadati di local_authority/charger_speed — in
    quel caso site_id="" e charging_mode="AC" (default) per ogni
    record. Per popolarli, chiamare load_metadata() prima, oppure usare
    la scorciatoia load_with_metadata().
    """

    FEATURE_NAMES = [
        "session_id",       # sintetizzato: hash(CPID, start_time) — vedi _synthesize_session_id
        "node_id",          # CPID (charge point ID)
        "cluster_id",       # non presente nel dataset → ""
        "site_id",          # local_authority (32 council area scozzesi), da metadati CPID
        "user_id",          # non tracciato da ChargePlace Scotland → sempre None
        "start_time",       # 'Start' (data) + 'Time' (ora) combinati
        "end_time",         # start_time + Duration
        "timezone",         # sempre "Europe/London"
        "done_charging_time",  # == end_time (nessun concetto distinto in questo dataset)
        "total_energy_kwh", # 'Consumed(kWh)'
        "max_power_kw",     # calcolato: energia / ore di ricarica
        "kwh_requested",    # non presente → 0.0
        "minutes_available",# non presente → 0
        "charging_mode",    # 'Connector_Type' da metadati CPID, default "AC"
        "temperature_c",    # non presente → None
        "error_code",       # non presente → None
        "anomaly_label",    # non etichettato → None
    ]

    def __init__(self):
        self._data: list[dict[str, Any]] = []
        self._local_authority: dict[str, str] = {}
        self._charger_speed: dict[str, str] = {}

    def load_metadata(self, metadata_dir: str) -> None:
        """
        Carica le mappe CPID -> local_authority e CPID -> Connector_Type
        da CPID_information/. Va chiamato PRIMA di load()/load_multiple()
        se si vuole site_id/charging_mode popolati correttamente
        (altrimenti restano rispettivamente "" e "AC" per ogni record).
        """
        meta_dir = Path(metadata_dir)
        la_path = meta_dir / "CPID_and_local_authority.xlsx"
        speed_path = meta_dir / "CPID_and_charger_speed.xlsx"

        if la_path.exists():
            df = pd.read_excel(la_path)
            self._local_authority = {
                str(row.CPID): str(row.local_authority) for row in df.itertuples()
            }
        else:
            logger.warning(f"Metadata local_authority non trovato: {la_path}")

        if speed_path.exists():
            df = pd.read_excel(speed_path)
            self._charger_speed = {
                str(row.CPID): str(row.Connector_Type) for row in df.itertuples()
            }
        else:
            logger.warning(f"Metadata charger_speed non trovato: {speed_path}")

    def load(self, path: str) -> None:
        """Carica un singolo file mensile di sessioni (contratto AbstractDataset)."""
        self._data = self._parse_file(path)

    def load_multiple(self, paths: list[str]) -> None:
        """Carica e concatena più file mensili."""
        self._data = []
        for path in paths:
            self._data.extend(self._parse_file(path))

    def load_with_metadata(self, session_paths: list[str], metadata_dir: str) -> None:
        """Scorciatoia: load_metadata() + load_multiple() in un solo passo."""
        self.load_metadata(metadata_dir)
        self.load_multiple(session_paths)

    def _parse_file(self, path: str) -> list[dict[str, Any]]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {path}")

        # Colonne rinominate subito dopo la lettura: "Consumed(kWh)" non è un
        # identificatore Python valido e renderebbe fragile/illeggibile
        # l'accesso via itertuples (nomi posizionali tipo "_2").
        df = pd.read_excel(file_path)
        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise KeyError(f"Colonne attese mancanti in {path}: {missing}")
        df = df.rename(columns={"Consumed(kWh)": "consumed_kwh"})

        records = []
        n_skipped = 0
        for row in df.itertuples(index=False):
            try:
                records.append(self._parse_record(row))
            except (ValueError, TypeError, AttributeError) as exc:
                n_skipped += 1
                logger.debug(f"Record scartato ({path}): {exc}")
        if n_skipped:
            logger.warning(f"Scartati {n_skipped}/{len(df)} record malformati da {path}")
        return records

    def _parse_record(self, row: Any) -> dict[str, Any]:
        cpid = str(row.CPID)

        # 'Start' è una data (mezzanotte), 'Time' è l'orario di inizio reale
        # nella stessa giornata — combiniamo i due per ottenere start_time.
        start_date = row.Start
        if pd.isna(start_date):
            raise ValueError("campo 'Start' mancante")
        time_val = row.Time
        if hasattr(time_val, "hour"):
            start = datetime.combine(start_date.date(), time_val)
        else:
            # fallback stringa "HH:MM:SS" (letture con altri motori)
            h, m, s = (int(p) for p in str(time_val).split(":")[:3])
            start = start_date.replace(hour=h, minute=m, second=s)

        duration = _parse_duration_to_timedelta(row.Duration)
        end = start + duration

        kwh = float(row.consumed_kwh) if not pd.isna(row.consumed_kwh) else 0.0

        # Fix 2026-09-14 (deep review round 3, bug reale trovato da subagent
        # + verificato leggendo scripts/run_experiments.py::enrich_sessions()):
        # 'start'/'end' qui sopra sono ora LOCALE della Scozia (wall-clock,
        # presi cosi' come sono scritti nelle celle Excel 'Start'/'Time'),
        # NON UTC — a differenza di src/adapters/acn_dataset.py, il cui
        # start_time/end_time SONO UTC nonostante il nome del campo (vedi
        # commento li' e la verifica empirica sul picco orario di office1).
        # enrich_sessions() applica pero' lo STESSO contratto a ogni dataset:
        # tratta sempre start_time come UTC e lo converte al fuso IANA del
        # campo 'timezone' per calcolare hour_of_day. Senza questa
        # conversione, un orario gia' locale veniva silenziosamente trattato
        # come UTC e poi "convertito" di nuovo a Europe/London: un no-op
        # durante l'ora solare GMT (UTC+0, nessun errore), ma uno sfasamento
        # sistematico di +1h durante l'ora legale BST (UTC+1, marzo-ottobre
        # circa) — hour_of_day errato per la maggioranza delle sessioni
        # estive. Fix: localizziamo qui start/end come Europe/London
        # (gestendo DST via zoneinfo) e li convertiamo a UTC prima di
        # salvarli, cosi' start_time/end_time rispettano lo stesso contratto
        # "sempre UTC" di ACN-Data e downstream non serve alcuna modifica.
        # _synthesize_session_id() continua a usare 'start' locale (non
        # UTC): e' solo un hash deterministico, non richiede un fuso
        # specifico, e cambiare l'input romperebbe la riproducibilita' di
        # session_id già eventualmente calcolati altrove senza alcun
        # beneficio.
        start_utc = start.replace(tzinfo=_LOCAL_TZ).astimezone(_UTC).replace(tzinfo=None)
        end_utc = end.replace(tzinfo=_LOCAL_TZ).astimezone(_UTC).replace(tzinfo=None)

        return {
            "session_id":           _synthesize_session_id(cpid, start),
            "node_id":              cpid,
            "cluster_id":           "",
            "site_id":              self._local_authority.get(cpid, ""),
            "user_id":              None,
            "start_time":           start_utc.isoformat(),
            "end_time":             end_utc.isoformat(),
            "timezone":             _TIMEZONE,
            "done_charging_time":   end_utc.isoformat(),
            "total_energy_kwh":     kwh,
            "max_power_kw":         _compute_max_power_kw(kwh, duration),
            "kwh_requested":        0.0,
            "minutes_available":    0,
            "charging_mode":        self._charger_speed.get(cpid, "AC"),
            "temperature_c":        None,
            "error_code":           None,
            "anomaly_label":        None,
        }

    def get_sample(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self._data):
            raise IndexError(
                f"Index {index} out of range (dataset size: {len(self._data)})"
            )
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)

    def get_feature_names(self) -> list[str]:
        return self.FEATURE_NAMES
