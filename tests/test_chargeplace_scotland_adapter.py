# tests/test_chargeplace_scotland_adapter.py
"""
Unit Tests — ChargePlaceScotlandDataset
========================================
Verifica il corretto funzionamento dell'adapter ChargePlaceScotlandDataset
sui file reali ChargePlace Scotland (task #37).

Cosa testiamo:
- Caricamento corretto dei file .xlsx mensili (con e senza metadati CPID)
- Numero totale di sessioni atteso (verificato manualmente coi file reali)
- Presenza di tutte le feature standard (stesso contratto di ACNDataset)
- Join site_id/charging_mode con i metadati CPID (copertura attesa ~99.9%)
- Campi assenti in questo dataset (user_id, kwh_requested, ecc.) → default
  documentati, mai silenziosamente sbagliati
- Gestione degli indici fuori range
- Parsing robusto di Duration (datetime.time da Excel)

Questo test NON tocca FL, protocolli o il Privacy Auditor.
"""

import pytest
from datetime import datetime, timedelta
from types import SimpleNamespace

from src.adapters.chargeplace_scotland_adapter import (
    ChargePlaceScotlandDataset,
    _compute_max_power_kw,
    _parse_duration_to_timedelta,
    _synthesize_session_id,
)

OCT_23 = "datasets/alt/chargeplace_scotland/Sessions_from_CPS/OCT-23.xlsx"
NOV_23 = "datasets/alt/chargeplace_scotland/Sessions_from_CPS/NOV-23.xlsx"
METADATA_DIR = "datasets/alt/chargeplace_scotland/CPID_information"


@pytest.fixture(scope="module")
def ds():
    """Fixture: carica OCT-23 con metadati CPID (site_id/charging_mode popolati).

    Fix 2026-09-14 (deep review round 3): era function-scoped, quindi
    ricaricava OCT-23.xlsx (162.896 righe, ~7.5s) da zero per OGNI test che
    la usa (19/30 test in questo file) — nessun test muta il dataset
    (verificato: nessuna scrittura su ds.*/ds_no_metadata.* in questo file),
    quindi scope="module" e' sicuro e riduce il tempo della suite da minuti
    a pochi secondi.
    """
    dataset = ChargePlaceScotlandDataset()
    dataset.load_with_metadata(session_paths=[OCT_23], metadata_dir=METADATA_DIR)
    return dataset


@pytest.fixture(scope="module")
def ds_no_metadata():
    """Fixture: carica OCT-23 SENZA metadati (contratto AbstractDataset puro).

    Stesso fix di `ds` sopra — module-scoped, nessun test muta il dataset.
    """
    dataset = ChargePlaceScotlandDataset()
    dataset.load(OCT_23)
    return dataset


# --- Test caricamento ---

def test_load_total_sessions(ds):
    """
    OCT-23.xlsx ha 162896 righe totali, di cui 7 con Duration NaN (verificato
    manualmente, 2026-09-09) — quelle 7 vengono scartate come record
    malformati, quindi il dataset caricato deve contenerne 162889.
    """
    assert len(ds) == 162896 - 7


def test_load_multiple_files():
    """Il caricamento combinato di due mesi deve sommare i conteggi individuali (al netto degli scarti)."""
    dataset = ChargePlaceScotlandDataset()
    dataset.load_with_metadata(session_paths=[OCT_23, NOV_23], metadata_dir=METADATA_DIR)
    assert len(dataset) == (162896 - 7) + (168780 - 6)


def test_load_single_file_without_metadata(ds_no_metadata):
    """load() (contratto AbstractDataset puro, senza metadati) deve funzionare senza errori."""
    assert len(ds_no_metadata) > 0


def test_file_not_found():
    """load() deve sollevare FileNotFoundError se il file non esiste."""
    dataset = ChargePlaceScotlandDataset()
    with pytest.raises(FileNotFoundError):
        dataset.load("datasets/alt/chargeplace_scotland/Sessions_from_CPS/nonexistent.xlsx")


# --- Test feature names ---

def test_feature_names_complete(ds):
    """Tutte le feature standard del framework devono essere presenti (stesso contratto di ACNDataset)."""
    names = ds.get_feature_names()
    expected = [
        "session_id", "node_id", "cluster_id", "site_id", "user_id",
        "start_time", "end_time", "timezone", "done_charging_time",
        "total_energy_kwh", "max_power_kw", "kwh_requested",
        "minutes_available", "charging_mode",
        "temperature_c", "error_code", "anomaly_label",
    ]
    for feature in expected:
        assert feature in names, f"Feature mancante: {feature}"


# --- Test struttura campione ---

def test_sample_has_all_features(ds):
    """Ogni campione deve contenere tutte le chiavi standard."""
    sample = ds.get_sample(0)
    for feature in ds.get_feature_names():
        assert feature in sample, f"Chiave mancante nel campione: {feature}"


def test_sample_session_id_is_string(ds):
    """session_id (sintetizzato) deve essere una stringa non vuota."""
    sample = ds.get_sample(0)
    assert isinstance(sample["session_id"], str)
    assert len(sample["session_id"]) > 0


def test_sample_energy_is_float(ds):
    """total_energy_kwh deve essere un float."""
    sample = ds.get_sample(0)
    assert isinstance(sample["total_energy_kwh"], float)


def test_sample_max_power_non_negative(ds):
    """max_power_kw deve essere >= 0.0 per le prime 200 sessioni."""
    for i in range(200):
        sample = ds.get_sample(i)
        assert sample["max_power_kw"] >= 0.0, f"max_power_kw negativo alla sessione {i}"


def test_sample_timezone_is_europe_london(ds):
    """timezone deve essere sempre 'Europe/London' (ChargePlace Scotland opera solo in Scozia)."""
    sample = ds.get_sample(0)
    assert sample["timezone"] == "Europe/London"


def test_done_charging_time_equals_end_time(ds):
    """Questo dataset non distingue fine-ricarica da disconnessione: done_charging_time == end_time."""
    sample = ds.get_sample(0)
    assert sample["done_charging_time"] == sample["end_time"]


# --- Test join metadati CPID ---

def test_site_id_populated_with_metadata(ds):
    """
    Con i metadati caricati, la stragrande maggioranza delle sessioni deve avere
    site_id (local_authority) non vuoto — copertura attesa ~99.9% (verificato
    manualmente: 162783/162896 su OCT-23).
    """
    n_populated = sum(1 for i in range(len(ds)) if ds.get_sample(i)["site_id"] != "")
    coverage = n_populated / len(ds)
    assert coverage > 0.99, f"Copertura site_id troppo bassa: {coverage:.4f}"


def test_site_id_empty_without_metadata(ds_no_metadata):
    """Senza load_metadata(), site_id deve restare '' (mai un default silenziosamente sbagliato)."""
    sample = ds_no_metadata.get_sample(0)
    assert sample["site_id"] == ""


def test_charging_mode_defaults_to_ac_without_metadata(ds_no_metadata):
    """Senza metadati charger_speed, charging_mode deve ricadere sul default 'AC'."""
    sample = ds_no_metadata.get_sample(0)
    assert sample["charging_mode"] == "AC"


# --- Test campi assenti in questo dataset → default documentati ---

def test_user_id_always_none(ds):
    """user_id non è tracciato da ChargePlace Scotland → sempre None (mai un ID inventato)."""
    for i in range(50):
        assert ds.get_sample(i)["user_id"] is None


def test_kwh_requested_always_zero(ds):
    """kwh_requested non esiste in questo dataset → sempre 0.0, mai confuso con un valore reale."""
    assert ds.get_sample(0)["kwh_requested"] == 0.0


def test_minutes_available_always_zero(ds):
    """minutes_available non esiste in questo dataset → sempre 0."""
    assert ds.get_sample(0)["minutes_available"] == 0


def test_missing_temperature_is_none(ds):
    """temperature_c non è presente → deve essere None, mai 0.0."""
    assert ds.get_sample(0)["temperature_c"] is None


def test_missing_anomaly_label_is_none(ds):
    """anomaly_label non è etichettato → deve essere None, mai 0."""
    assert ds.get_sample(0)["anomaly_label"] is None


# --- Test conversione UTC (fix 2026-09-14, deep review round 3/4) ---
#
# _parse_record() combina 'Start'+'Time' in un orario LOCALE Europe/London
# (cosi' come scritto nelle celle Excel), poi lo converte a UTC prima di
# salvarlo in start_time/end_time/done_charging_time — stesso contratto di
# acn_dataset.py, da cui dipende scripts/run_experiments.py::enrich_sessions()
# per calcolare hour_of_day. Prima del fix, start_time era lasciato in ora
# locale grezza: enrich_sessions() lo trattava comunque come UTC e lo
# "riconvertiva" a Europe/London, sfasando hour_of_day di +1h durante l'ora
# legale BST (nessun errore in ora solare GMT, dove l'offset locale è 0).
# Questi test isolano _parse_record() con una riga sintetica (bypassando il
# caricamento di un intero file .xlsx) per verificare la conversione stessa,
# non solo il suo effetto a valle — gap segnalato esplicitamente dal deep
# review round 4 (nessun test esistente copriva la conversione).

def _make_row(cpid="12345", start_date=None, time_val=None, duration=None, kwh=1.0):
    """Riga sintetica compatibile con l'interfaccia usata da _parse_record()
    (attributi CPID/Start/Time/Duration/consumed_kwh, come da itertuples())."""
    from datetime import time as _time
    if start_date is None:
        start_date = datetime(2023, 6, 15)  # BST di default
    if time_val is None:
        time_val = _time(14, 30, 0)
    if duration is None:
        duration = timedelta(hours=1)
    return SimpleNamespace(
        CPID=cpid, Start=start_date, Time=time_val, Duration=duration, consumed_kwh=kwh,
    )


def test_parse_record_converts_bst_local_time_to_utc():
    """15 giugno 2023, 14:30 locale (BST, UTC+1) deve diventare 13:30 UTC."""
    ds = ChargePlaceScotlandDataset()
    row = _make_row(start_date=datetime(2023, 6, 15))
    record = ds._parse_record(row)
    assert record["start_time"] == "2023-06-15T13:30:00"


def test_parse_record_leaves_gmt_local_time_unchanged():
    """15 dicembre 2023, 14:30 locale (GMT, UTC+0) resta 14:30 UTC (nessuno sfasamento)."""
    ds = ChargePlaceScotlandDataset()
    row = _make_row(start_date=datetime(2023, 12, 15))
    record = ds._parse_record(row)
    assert record["start_time"] == "2023-12-15T14:30:00"


def test_parse_record_end_time_and_done_charging_time_also_converted_to_utc():
    """end_time e done_charging_time devono riflettere la stessa conversione UTC di start_time (bug trovato: prima del fix done_charging_time restava in ora locale anche dopo il primo fix su start_time)."""
    ds = ChargePlaceScotlandDataset()
    row = _make_row(start_date=datetime(2023, 6, 15), duration=timedelta(hours=2))
    record = ds._parse_record(row)
    assert record["end_time"] == "2023-06-15T15:30:00"
    assert record["done_charging_time"] == record["end_time"]


def test_parse_record_hour_of_day_matches_true_local_hour_post_fix():
    """
    Verifica end-to-end del motivo del fix: applicando la stessa logica di
    scripts/run_experiments.py::enrich_sessions() (start_time trattato come
    UTC, poi convertito al fuso 'timezone') si deve ottenere l'ora locale
    ORIGINALE (14), non 15 (il bug pre-fix) — sia in BST che in GMT.
    """
    from zoneinfo import ZoneInfo
    ds = ChargePlaceScotlandDataset()
    for start_date in (datetime(2023, 6, 15), datetime(2023, 12, 15)):
        row = _make_row(start_date=start_date)
        record = ds._parse_record(row)
        parsed_utc = datetime.fromisoformat(record["start_time"]).replace(tzinfo=ZoneInfo("UTC"))
        local = parsed_utc.astimezone(ZoneInfo(record["timezone"]))
        assert local.hour == 14, f"hour_of_day errato per {start_date}: {local.hour}"


# --- Test indici ---

def test_index_out_of_range(ds):
    """get_sample() deve sollevare IndexError per indici fuori range."""
    with pytest.raises(IndexError):
        ds.get_sample(99999999)


def test_negative_index_raises(ds):
    """get_sample() deve sollevare IndexError anche per indici negativi."""
    with pytest.raises(IndexError):
        ds.get_sample(-1)


# --- Test funzioni di utilità ---

def test_compute_max_power_kw_correct():
    """Con 10 kWh erogati in 2 ore -> potenza media = 5.0 kW."""
    result = _compute_max_power_kw(10.0, timedelta(hours=2))
    assert result == 5.0


def test_compute_max_power_zero_duration():
    """Durata zero -> deve restituire 0.0, non sollevare ZeroDivisionError."""
    result = _compute_max_power_kw(10.0, timedelta(seconds=0))
    assert result == 0.0


def test_parse_duration_from_time_object():
    """datetime.time (formato tipico letto da Excel per durate < 24h)."""
    from datetime import time
    result = _parse_duration_to_timedelta(time(1, 30, 45))
    assert result == timedelta(hours=1, minutes=30, seconds=45)


def test_parse_duration_from_string_fallback():
    """Stringa 'HH:MM:SS' (fallback per letture con altri motori)."""
    result = _parse_duration_to_timedelta("02:15:00")
    assert result == timedelta(hours=2, minutes=15)


def test_parse_duration_from_timedelta_passthrough():
    """Un pandas.Timedelta / datetime.timedelta già parsato deve passare invariato."""
    td = timedelta(hours=5, minutes=10)
    assert _parse_duration_to_timedelta(td) == td


def test_parse_duration_over_24h_singular_day():
    """
    Bug reale trovato scrivendo questi test (2026-09-09): le sessioni >= 24h
    arrivano come stringa 'N day(s), H:MM:SS' (str(timedelta) nativo), non
    come 'HH:MM:SS' — un semplice split(':') falliva su queste 993/162896
    sessioni di OCT-23.xlsx. NON è un troncamento modulo 24h: il valore è
    corretto e completo.
    """
    result = _parse_duration_to_timedelta("1 day, 8:37:17")
    assert result == timedelta(days=1, hours=8, minutes=37, seconds=17)


def test_parse_duration_over_24h_plural_days():
    result = _parse_duration_to_timedelta("2 days, 17:15:00")
    assert result == timedelta(days=2, hours=17, minutes=15)


def test_synthesize_session_id_deterministic():
    """Lo stesso (CPID, start_time) deve produrre sempre lo stesso session_id sintetizzato."""
    start = datetime(2023, 10, 31, 23, 10, 55)
    id1 = _synthesize_session_id("52935", start)
    id2 = _synthesize_session_id("52935", start)
    assert id1 == id2


def test_synthesize_session_id_differs_by_cpid():
    """CPID diversi con lo stesso start_time devono produrre session_id diversi."""
    start = datetime(2023, 10, 31, 23, 10, 55)
    id1 = _synthesize_session_id("52935", start)
    id2 = _synthesize_session_id("51429", start)
    assert id1 != id2
