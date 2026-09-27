"""
Test dell'etichetta di cella (scripts/etichetta_cella.py) e della sua adozione in
check_significance.discover_groups: segnalazioni 44, 45, 53, 2026-09-24.
Nessuna dipendenza torch.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_significance as cs  # noqa: E402
from etichetta_cella import etichetta_cella  # noqa: E402


def _cfg(**kw):
    c = {"dp_mode": "dp-fedavg", "epsilon": 1.0, "no_dp": True, "delta": 1e-5,
         "fl_rounds": 10, "proximal_mu": 0.01, "epochs": 50, "hidden_dims": None,
         "latent_dim": 4, "feature_names": None, "seed": 42}
    c.update(kw)
    return c


def test_valori_di_base_danno_l_etichetta_storica():
    assert etichetta_cella(_cfg()) == "no-DP baseline"
    assert etichetta_cella(_cfg(no_dp=False, epsilon=16.0)) == "dp-fedavg, eps=16.0"
    assert etichetta_cella(_cfg(no_dp=False, epsilon=16.0, max_grad_norm=1.0,
                                common_init=False, partition_strategy="per_site",
                                split_strategy="random")) == "dp-fedavg, eps=16.0"
    rdp = _cfg(record_dp={"enabled": True, "noise_multiplier": 1.0})
    assert etichetta_cella(rdp) == "record-DP, nm=1.0"


def test_i_trattamenti_diversi_hanno_celle_diverse():
    assert etichetta_cella(_cfg(no_dp=False, epsilon=16.0, max_grad_norm=0.25)) \
        == "dp-fedavg, eps=16.0, C=0.25"
    assert etichetta_cella(_cfg(common_init=True)) == "no-DP baseline, init comune"
    assert etichetta_cella(_cfg(proximal_mu=0.0)) == "no-DP baseline, mu=0.0"
    assert etichetta_cella(_cfg(partition_strategy="iid")) == "no-DP baseline, partizione=iid"
    assert etichetta_cella(_cfg(split_strategy="entity_aware")) == "no-DP baseline, split=entity_aware"
    assert etichetta_cella(_cfg(epochs=100, fl_rounds=3)) == "no-DP baseline, epoche=100, round=3"


def test_c_non_conta_senza_clipping_client_level():
    assert etichetta_cella(_cfg(max_grad_norm=0.25)) == "no-DP baseline"


def test_campi_assenti_o_equivalenti_non_aggiungono_suffissi():
    c = _cfg()
    for k in ("fl_rounds", "proximal_mu", "epochs", "hidden_dims", "latent_dim", "feature_names"):
        c.pop(k)
    assert etichetta_cella(c) == "no-DP baseline"
    assert etichetta_cella(_cfg(hidden_dims=[16, 8])) == "no-DP baseline"
    std = ["total_energy_kwh", "max_power_kw", "kwh_requested",
           "minutes_available", "hour_of_day", "duration_hours"]
    assert etichetta_cella(_cfg(feature_names=std)) == "no-DP baseline"
    assert etichetta_cella(_cfg(feature_names=std + ["x"])) == "no-DP baseline, feature=7"


def _scrivi(path: Path, cfg: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": cfg, "summary": {"mean_lira_auc_roc": 0.5}}))


def test_run_con_c_diverso_non_sostituisce_il_seed_di_e_a(tmp_path):
    # regressione della segnalazione 53
    _scrivi(tmp_path / "rq1-eps16" / "experiment_20260923_183148.json",
            _cfg(no_dp=False, epsilon=16.0))
    _scrivi(tmp_path / "op-C0.25-eps16" / "experiment_20260925_010000.json",
            _cfg(no_dp=False, epsilon=16.0, max_grad_norm=0.25))
    g = cs.discover_groups(pattern=str(tmp_path / "*/experiment_*.json"))
    assert len(g["dp-fedavg, eps=16.0"]) == 1
    assert "rq1-eps16" in g["dp-fedavg, eps=16.0"][0]
    assert len(g["dp-fedavg, eps=16.0, C=0.25"]) == 1


def test_varianti_no_dp_non_sostituiscono_il_riferimento(tmp_path):
    # regressione della segnalazione 45 (E-D, E-E) e della 53 (common_init)
    _scrivi(tmp_path / "nodp-sweep2" / "experiment_20260908_121719.json", _cfg())
    _scrivi(tmp_path / "rq3-mu0" / "experiment_20260925_120000.json", _cfg(proximal_mu=0.0))
    _scrivi(tmp_path / "rq2-iid" / "experiment_20260923_120000.json",
            _cfg(partition_strategy="iid"))
    _scrivi(tmp_path / "ctrl-init" / "experiment_20260924_164941.json", _cfg(common_init=True))
    g = cs.discover_groups(pattern=str(tmp_path / "*/experiment_*.json"))
    assert g["no-DP baseline"] == [str(tmp_path / "nodp-sweep2" / "experiment_20260908_121719.json")]
    assert set(g) == {"no-DP baseline", "no-DP baseline, mu=0.0",
                      "no-DP baseline, partizione=iid", "no-DP baseline, init comune"}
