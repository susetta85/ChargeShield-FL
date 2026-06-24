# src/adapters/acn_dataset.py
"""
ACN-Data Dataset Adapter
========================
Traduce il dataset pubblico ACN-Data (Adaptive Charging Network, Caltech)
dal formato JSON grezzo al formato standard del framework ChargeShield-FL.

Dataset source: https://ev.caltech.edu/dataset
Siti supportati: JPL, Caltech
Formato input: JSON con chiavi _meta e _items

Responsabilità di questo modulo:
- Caricare il file JSON da disco
- Mappare i campi ACN ai campi standard del framework
- Calcolare max_power_kw dalla durata della sessione
- Gestire i campi assenti con valori di default sicuri
- Esporre i dati tramite l'interfaccia AbstractDataset

Questo adapter NON conosce FL, protocolli, o il Privacy Auditor.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.base_dataset import AbstractDataset


# Formato data usato da ACN-Data nei campi connectionTime / disconnectTime
_ACN_DATE_FORMAT = "%a, %d %b %Y %H:%M:%S %Z"


def _parse_acn_datetime(value: str) -> datetime:
    """
    Converte una stringa data ACN-Data in oggetto datetime.
    Esempio input: 'Wed, 01 Jan 2020 18:10:34 GMT'
    """
    return datetime.strptime(value, _ACN_DATE_FORMAT)


def _compute_max_power_kw(kwh: float, start: datetime, end: datetime) -> float:
    """
    Stima la potenza massima media in kW.
    Formula: kWhDelivered / ore_di_ricarica_effettiva.
    Restituisce 0.0 se la durata è zero o negativa (sessione anomala).
    """
    duration_hours = (end - start).total_seconds() / 3600.0
    if duration_hours <= 0:
        return 0.0
    return round(kwh / duration_hours, 3)


class ACNDataset(AbstractDataset):
    """
    Adapter per il dataset ACN-Data in formato JSON.

    Uso tipico:
        ds = ACNDataset()
        ds.load("datasets/acn/jpl/acndata_sessions 2020.json")
        sample = ds.get_sample(0)
        print(sample["total_energy_kwh"])
    """

    # Nomi delle feature standard esposte dal framework.
    # Questi nomi sono il contratto con il resto del sistema.
    FEATURE_NAMES = [
        "session_id",       # ID univoco della sessione di ricarica
        "node_id",          # ID della stazione (stationID in ACN)
        "cluster_id",       # ID del cluster (clusterID in ACN)
        "site_id",          # Sito fisico: JPL o Caltech
        "user_id",          # ID anonimizzato dell'utente
        "start_time",       # Orario di connessione del veicolo
        "end_time",         # Orario di disconnessione del veicolo
        "done_charging_time",  # Orario fine ricarica effettiva
        "total_energy_kwh", # Energia erogata in kWh
        "max_power_kw",     # Potenza media stimata in kW
        "kwh_requested",    # Energia richiesta dall'utente (userInputs)
        "minutes_available",# Minuti disponibili dichiarati dall'utente
        "charging_mode",    # AC (default, JPL non lo specifica)
        "temperature_c",    # Temperatura (non presente in ACN → 0.0)
        "error_code",       # Codice errore (non presente in ACN → None)
        "anomaly_label",    # Etichetta anomalia (0=normale, 1=anomalia)
    ]

    def __init__(self):
        # Lista interna dei record già parsati e normalizzati
        self._data: list[dict[str, Any]] = []

    def load(self, path: str) -> None:
        """
        Carica il file JSON di ACN-Data da disco e popola _data.
        Ogni elemento di _items viene convertito in un dict standard.

        Args:
            path: percorso al file JSON (es. datasets/acn/jpl/acndata_sessions 2020.json)

        Raises:
            FileNotFoundError: se il file non esiste
            KeyError: se il JSON non ha la struttura attesa (_items)
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {path}")

        with open(file_path, encoding="utf-8") as f:
            raw = json.load(f)

        # ACN-Data usa _items come lista delle sessioni
        items = raw["_items"]
        self._data = [self._parse_record(item) for item in items]

    def load_multiple(self, paths: list[str]) -> None:
        """
        Carica e concatena più file JSON ACN-Data.
        Utile per combinare anni diversi (2019 + 2020).
        """
        self._data = []
        for path in paths:
            file_path = Path(path)
            if not file_path.exists():
                raise FileNotFoundError(f"Dataset not found at: {path}")
            with open(file_path, encoding="utf-8") as f:
                raw = json.load(f)
            self._data.extend(
                [self._parse_record(item) for item in raw["_items"]]
            )

    def _parse_record(self, item: dict[str, Any]) -> dict[str, Any]:
        """
        Converte un singolo record ACN-Data nel formato standard.
        Gestisce i campi mancanti con valori di default sicuri.

        userInputs è una lista — prendiamo il primo elemento se esiste,
        altrimenti usiamo valori di default.
        """
        # Parsing date
        start = _parse_acn_datetime(item["connectionTime"])
        end = _parse_acn_datetime(item["disconnectTime"])
        done_raw = item.get("doneChargingTime") or item.get("disconnectTime")
        done = _parse_acn_datetime(done_raw)

        kwh = float(item.get("kWhDelivered", 0.0))

        # userInputs è opzionale — non tutte le sessioni hanno input utente
        user_inputs = item.get("userInputs", [])
        ui = user_inputs[0] if user_inputs else {}

        return {
            "session_id":        item.get("sessionID", ""),
            "node_id":           item.get("stationID", ""),
            "cluster_id":        item.get("clusterID", ""),
            "site_id":           item.get("siteID", ""),
            "user_id":           item.get("userID", ""),
            "start_time":        start.isoformat(),
            "end_time":          end.isoformat(),
            "done_charging_time": done.isoformat(),
            "total_energy_kwh":  kwh,
            # Calcolato: energia erogata / ore di ricarica effettiva
            "max_power_kw":      _compute_max_power_kw(kwh, start, done),
            # Da userInputs — energia richiesta dall'utente
            "kwh_requested":     float(ui.get("kWhRequested", 0.0)),
            # Da userInputs — minuti disponibili dichiarati
            "minutes_available": int(ui.get("minutesAvailable", 0)),
            # JPL non specifica il modo → AC per default
            "charging_mode":     "AC",
            # Non presente in ACN-Data
            "temperature_c":     None,
            "error_code":        None,
            # Non presente → da inferire
            "anomaly_label":     None,
        }

    def get_sample(self, index: int) -> dict[str, Any]:
        """
        Restituisce un singolo campione per indice.

        Raises:
            IndexError: se l'indice è fuori range
        """
        if index < 0 or index >= len(self._data):
            raise IndexError(
                f"Index {index} out of range (dataset size: {len(self._data)})"
            )
        return self._data[index]

    def __len__(self) -> int:
        """Numero totale di sessioni caricate."""
        return len(self._data)

    def get_feature_names(self) -> list[str]:
        """Restituisce i nomi delle feature standard del framework."""
        return self.FEATURE_NAMES
