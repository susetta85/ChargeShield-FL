"""
Test puro-Python (nessuna dipendenza torch/nvflare, entrambi assenti in
questo sandbox — stesso limite di tests/test_fedmia_gradient_pairing.py e
tests/test_canary_group_sampling.py) per il fix Sprint 10zz+76:
chargeshield_executor.py::_enrich_sessions() mancava hour_of_day_sin/
hour_of_day_cos/start_time_epoch, presenti invece nella funzione canonica
scripts/run_experiments.py::enrich_sessions() — trovato dalla deep review
round 5 (README Sprint 10zz+74).

Replica ESATTA della logica di ENTRAMBE le funzioni (non importabili qui:
chargeshield_executor.py importa nvflare a livello di modulo,
run_experiments.py importa torch) — se una delle due cambia, aggiornare
insieme qui. Lo scopo di questo file è verificare che le due copie
producano ORA lo stesso output sullo stesso input, non solo che
_enrich_sessions() in isolamento sia "corretta".

Nota (trovata da deep review round 6, 2026-09-14): ENTRAMBE le funzioni
reali mutano i dict di sessione IN PLACE (nessuna copia difensiva) — le
due repliche qui sotto devono fare lo stesso per restare fedeli. I test
di parità (_produce_identical_output_*) per questo passano SEMPRE una
lista di sessioni fresca e indipendente a ciascuna chiamata (mai la
stessa lista/stessi dict condivisi fra le due funzioni): con la mutazione
in-place, riusare gli stessi oggetti renderebbe `a == b` banalmente vero
per identità, non un confronto reale della logica.
"""
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest


def _enrich_sessions_executor(sessions):
    """Replica esatta, POST-fix Sprint 10zz+76, di
    chargeshield_executor.py::_enrich_sessions()."""
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
                    hour_of_day = float(start.hour)
            else:
                hour_of_day = float(start.hour)

            s["hour_of_day"] = hour_of_day
            s["hour_of_day_sin"] = math.sin(2.0 * math.pi * hour_of_day / 24.0)
            s["hour_of_day_cos"] = math.cos(2.0 * math.pi * hour_of_day / 24.0)
            s["duration_hours"] = max(0.0, (end - start).total_seconds() / 3600.0)
            s["start_time_epoch"] = start.replace(tzinfo=ZoneInfo("UTC")).timestamp()
            enriched.append(s)
        except (KeyError, ValueError):
            pass
    return enriched


def _enrich_sessions_canonical(sessions):
    """Replica esatta di scripts/run_experiments.py::enrich_sessions() —
    stessa funzione da cui il fix Sprint 10zz+76 e' stato copiato parola
    per parola nell'executor."""
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
                    hour_of_day = float(start.hour)
            else:
                hour_of_day = float(start.hour)

            s["hour_of_day"] = hour_of_day
            s["hour_of_day_sin"] = math.sin(2.0 * math.pi * hour_of_day / 24.0)
            s["hour_of_day_cos"] = math.cos(2.0 * math.pi * hour_of_day / 24.0)
            s["duration_hours"] = max(0.0, (end - start).total_seconds() / 3600.0)
            s["start_time_epoch"] = start.replace(tzinfo=ZoneInfo("UTC")).timestamp()
            enriched.append(s)
        except (KeyError, ValueError):
            pass
    return enriched


def _mk(start, end, timezone=None):
    return {"start_time": start, "end_time": end, "timezone": timezone}


# ── Le due copie ora producono output identico (obiettivo del fix) ─────────────

def test_executor_and_canonical_produce_identical_output_no_timezone():
    # Liste INDIPENDENTI per ciascuna chiamata (entrambe le funzioni mutano
    # i dict in place, come le funzioni reali — vedi nota nel docstring del
    # modulo): condividere gli stessi dict renderebbe il confronto banale.
    a = _enrich_sessions_executor([_mk("2023-06-15T14:30:00", "2023-06-15T16:00:00")])
    b = _enrich_sessions_canonical([_mk("2023-06-15T14:30:00", "2023-06-15T16:00:00")])
    assert a == b


def test_executor_and_canonical_produce_identical_output_with_timezone():
    a = _enrich_sessions_executor(
        [_mk("2023-06-15T14:30:00", "2023-06-15T16:00:00", timezone="America/Los_Angeles")]
    )
    b = _enrich_sessions_canonical(
        [_mk("2023-06-15T14:30:00", "2023-06-15T16:00:00", timezone="America/Los_Angeles")]
    )
    assert a == b


def test_executor_and_canonical_identical_across_multiple_sites_and_hours():
    def _sessions():
        return [
            _mk("2023-01-01T00:00:00", "2023-01-01T01:00:00", timezone="America/Los_Angeles"),
            _mk("2023-06-15T23:45:00", "2023-06-16T02:00:00", timezone="America/New_York"),
            _mk("2023-03-10T06:00:00", "2023-03-10T06:30:00"),  # nessun timezone -> fallback
        ]
    a = _enrich_sessions_executor(_sessions())
    b = _enrich_sessions_canonical(_sessions())
    assert a == b


# ── Le 3 feature mancanti (gap Sprint 10zz+74) ora sono presenti nell'executor ──

def test_hour_of_day_sin_cos_present_and_correct():
    sessions = [_mk("2023-06-15T06:00:00", "2023-06-15T07:00:00")]
    out = _enrich_sessions_executor(sessions)
    assert len(out) == 1
    s = out[0]
    assert "hour_of_day_sin" in s and "hour_of_day_cos" in s
    # hour_of_day=6 -> angolo = pi/2 -> sin=1, cos=0
    assert s["hour_of_day_sin"] == pytest.approx(1.0, abs=1e-9)
    assert s["hour_of_day_cos"] == pytest.approx(0.0, abs=1e-9)


def test_hour_of_day_sin_cos_unit_circle_for_every_hour():
    # sin^2 + cos^2 = 1 per ogni ora intera 0-23 (proprietà della codifica
    # circolare, indipendente dal valore specifico di hour_of_day).
    for h in range(24):
        sessions = [_mk(f"2023-06-15T{h:02d}:00:00", f"2023-06-15T{h:02d}:30:00")]
        out = _enrich_sessions_executor(sessions)
        s = out[0]
        norm = s["hour_of_day_sin"] ** 2 + s["hour_of_day_cos"] ** 2
        assert norm == pytest.approx(1.0, abs=1e-9)


def test_hour_of_day_23_and_0_are_close_in_circular_encoding():
    # Il punto dell'encoding circolare (Fase 8): 23h e 0h devono essere
    # vicine nello spazio (sin, cos), a differenza di hour_of_day lineare
    # (23 vs 0, la distanza massima possibile).
    s23 = _enrich_sessions_executor([_mk("2023-06-15T23:00:00", "2023-06-15T23:30:00")])[0]
    s00 = _enrich_sessions_executor([_mk("2023-06-16T00:00:00", "2023-06-16T00:30:00")])[0]
    dist = math.hypot(s23["hour_of_day_sin"] - s00["hour_of_day_sin"],
                       s23["hour_of_day_cos"] - s00["hour_of_day_cos"])
    # Un'ora di differenza sul cerchio a 24 punti: corda attesa piccola,
    # ben lontana dalla distanza massima possibile (diametro = 2.0).
    assert dist < 0.3


def test_start_time_epoch_present_and_matches_utc_timestamp():
    sessions = [_mk("2023-06-15T14:30:00", "2023-06-15T16:00:00")]
    out = _enrich_sessions_executor(sessions)
    s = out[0]
    assert "start_time_epoch" in s
    expected = datetime.fromisoformat("2023-06-15T14:30:00").replace(
        tzinfo=ZoneInfo("UTC")
    ).timestamp()
    assert s["start_time_epoch"] == pytest.approx(expected, abs=1e-6)


def test_start_time_epoch_distinct_for_distinct_sessions_same_day():
    # Quasi-univoca per sessione (razionale originale, Sprint 10kk) — due
    # sessioni diverse nello stesso giorno devono avere epoch diversi.
    s1 = _enrich_sessions_executor([_mk("2023-06-15T08:00:00", "2023-06-15T09:00:00")])[0]
    s2 = _enrich_sessions_executor([_mk("2023-06-15T08:00:01", "2023-06-15T09:00:00")])[0]
    assert s1["start_time_epoch"] != s2["start_time_epoch"]


# ── Comportamento pre-esistente invariato (regressione) ─────────────────────────

def test_hour_of_day_still_localized_by_timezone_unaffected_by_fix():
    # Stesso comportamento di prima del fix: hour_of_day resta calcolato in
    # ora locale del sito, non toccato dall'aggiunta delle 3 nuove feature.
    sessions = [_mk("2023-06-15T20:00:00", "2023-06-15T21:00:00", timezone="America/Los_Angeles")]
    out = _enrich_sessions_executor(sessions)
    # 20:00 UTC in giugno (PDT, UTC-7) -> 13:00 locale
    assert out[0]["hour_of_day"] == pytest.approx(13.0)


def test_malformed_timestamp_still_discarded_silently():
    # Comportamento invariato: eccezioni KeyError/ValueError scartano la
    # sessione senza sollevare, esattamente come prima del fix.
    sessions = [
        {"start_time": "not-a-date", "end_time": "2023-06-15T16:00:00"},
        {"end_time": "2023-06-15T16:00:00"},  # start_time mancante -> KeyError
        _mk("2023-06-15T14:30:00", "2023-06-15T16:00:00"),  # valida
    ]
    out = _enrich_sessions_executor(sessions)
    assert len(out) == 1
    assert out[0]["start_time"] == "2023-06-15T14:30:00"


def test_unknown_timezone_falls_back_to_raw_hour_not_discarded():
    sessions = [_mk("2023-06-15T14:30:00", "2023-06-15T16:00:00", timezone="Not/A_Real_Zone")]
    out = _enrich_sessions_executor(sessions)
    assert len(out) == 1
    assert out[0]["hour_of_day"] == 14.0
    # Anche nel fallback, le 3 feature devono comunque essere presenti.
    assert "hour_of_day_sin" in out[0]
    assert "start_time_epoch" in out[0]
