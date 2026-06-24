# tests/test_acn_dataset.py
"""
Unit Tests — ACNDataset
=======================
Verifica il corretto funzionamento dell'adapter ACNDataset
sul dataset reale ACN-Data JPL 2019+2020.

Cosa testiamo:
- Caricamento corretto dei file JSON
- Numero totale di sessioni atteso
- Presenza di tutte le feature standard
- Tipi dei campi (float, str, None)
- Calcolo di max_power_kw
- Gestione degli indici fuori range
- Campi assenti → None (mai default silenzioso)

Questo test NON tocca FL, protocolli o il Privacy Auditor.
"""

import pytest
from datetime import datetime

from src.adapters.acn_dataset import ACNDataset, _compute_max_power_kw


# Percorsi ai file JSON reali del dataset ACN-Data JPL
JPL_2019 = "datasets/acn/jpl/acndata_sessions_2019.json"
JPL_2020 = "datasets/acn/jpl/acndata_sessions_2020.json"


@pytest.fixture
def ds():
    """
    Fixture pytest: carica il dataset reale JPL 2019+2020.
    Riusata da tutti i test che richiedono dati reali.
    """
    dataset = ACNDataset()
    dataset.load_multiple([JPL_2019, JPL_2020])
    return dataset


# --- Test caricamento ---

def test_load_total_sessions(ds):
    """Il dataset combinato 2019+2020 deve contenere esattamente 13073 sessioni."""
    assert len(ds) == 13073


def test_load_single_file():
    """Il caricamento di un singolo file deve funzionare senza errori."""
    dataset = ACNDataset()
    dataset.load(JPL_2019)
    assert len(dataset) > 0


def test_file_not_found():
    """load() deve sollevare FileNotFoundError se il file non esiste."""
    dataset = ACNDataset()
    with pytest.raises(FileNotFoundError):
        dataset.load("datasets/acn/jpl/nonexistent.json")


# --- Test feature names ---

def test_feature_names_complete(ds):
    """Tutte le feature standard del framework devono essere presenti."""
    names = ds.get_feature_names()
    expected = [
        "session_id", "node_id", "cluster_id", "site_id", "user_id",
        "start_time", "end_time", "done_charging_time",
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
    """session_id deve essere una stringa non vuota."""
    sample = ds.get_sample(0)
    assert isinstance(sample["session_id"], str)
    assert len(sample["session_id"]) > 0


def test_sample_energy_is_float(ds):
    """total_energy_kwh deve essere un float."""
    sample = ds.get_sample(0)
    assert isinstance(sample["total_energy_kwh"], float)


def test_sample_max_power_non_negative(ds):
    """
    max_power_kw deve essere >= 0.0 per le prime 100 sessioni.
    Un valore negativo indicherebbe un errore nel calcolo della durata.
    """
    for i in range(100):
        sample = ds.get_sample(i)
        assert sample["max_power_kw"] >= 0.0, \
            f"max_power_kw negativo alla sessione {i}"


def test_sample_charging_mode_is_ac(ds):
    """charging_mode deve essere 'AC' — JPL non specifica il modo, default AC."""
    sample = ds.get_sample(0)
    assert sample["charging_mode"] == "AC"


# --- Test campi assenti → None ---

def test_missing_temperature_is_none(ds):
    """temperature_c non è presente in ACN-Data → deve essere None, mai 0.0."""
    sample = ds.get_sample(0)
    assert sample["temperature_c"] is None


def test_missing_error_code_is_none(ds):
    """error_code non è presente in ACN-Data → deve essere None."""
    sample = ds.get_sample(0)
    assert sample["error_code"] is None


def test_missing_anomaly_label_is_none(ds):
    """anomaly_label non è etichettato in ACN-Data → deve essere None, mai 0."""
    sample = ds.get_sample(0)
    assert sample["anomaly_label"] is None


# --- Test indici ---

def test_index_out_of_range(ds):
    """get_sample() deve sollevare IndexError per indici fuori range."""
    with pytest.raises(IndexError):
        ds.get_sample(99999)


def test_negative_index_raises(ds):
    """get_sample() deve sollevare IndexError anche per indici negativi."""
    with pytest.raises(IndexError):
        ds.get_sample(-1)


# --- Test funzioni di utilità ---

def test_compute_max_power_kw_correct():
    """
    Con 10 kWh erogati in 2 ore → potenza media = 5.0 kW.
    Verifica la formula base del calcolo.
    """
    start = datetime(2020, 1, 1, 8, 0, 0)
    end = datetime(2020, 1, 1, 10, 0, 0)
    result = _compute_max_power_kw(10.0, start, end)
    assert result == 5.0


def test_compute_max_power_zero_duration():
    """
    Se start == end la durata è zero → deve restituire 0.0, non sollevare ZeroDivisionError.
    """
    start = datetime(2020, 1, 1, 8, 0, 0)
    result = _compute_max_power_kw(10.0, start, start)
    assert result == 0.0
