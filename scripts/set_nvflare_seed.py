"""
Scrive config_fed_client.json + config_fed_server.json (dataset ACN, non
ChargePlace Scotland) per un dp_mode/seed specifico della campagna NVFLARE,
poi salva una copia del client config in seed_snapshots/ con un nome
univoco per dp_mode+seed — cosi' scripts/run_nvflare_mia.py puo' sempre
puntare allo snapshot giusto invece del file "live" (che viene
sovrascritto ad ogni run), evitando il bug gia' scoperto una volta
(seed123 rianalizzato con lo snapshot sbagliato).

Genera i file via json.dump (niente edit manuale in vim) — evita sia il
rischio di salvare JSON non valido sia la confusione vista con lo swap
file orfano di vim su questo stesso file.

Uso:
    python3 scripts/set_nvflare_seed.py --dp-mode central --seed 123
    python3 scripts/set_nvflare_seed.py --dp-mode local --seed 42

Dopo aver lanciato questo script, il passo successivo resta manuale:
    bash /workspace/startup/fl_admin.sh   # login admin@chargeshield.local
    submit_job /workspace/jobs/chargeshield_poc
"""

import argparse
import json
from pathlib import Path

CONFIG_DIR = Path("nvflare/jobs/chargeshield_poc/app/config")
CLIENT_PATH = CONFIG_DIR / "config_fed_client.json"
SERVER_PATH = CONFIG_DIR / "config_fed_server.json"
SNAPSHOT_DIR = CONFIG_DIR / "seed_snapshots"

VALID_MODES = ("dp-fedavg", "central", "local")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dp-mode", required=True, choices=VALID_MODES)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--epsilon", type=float, default=1.0)
    args = p.parse_args()

    with open(CLIENT_PATH) as f:
        client_cfg = json.load(f)
    with open(SERVER_PATH) as f:
        server_cfg = json.load(f)

    client_args = client_cfg["executors"][0]["executor"]["args"]
    # Riporta il client alla config ACN standard (rimuove eventuali chiavi
    # ChargePlace Scotland/canary lasciate da un run precedente — FIX
    # 2026-09-11, bug reale trovato durante il check richiesto dall'utente:
    # "canary" non veniva ripulito come dataset_adapter/chargeplace_scotland,
    # quindi un run di test col canary abilitato avrebbe silenziosamente
    # contaminato ogni run "pulito" successivo lanciato con questo script).
    client_args.pop("dataset_adapter", None)
    client_args.pop("chargeplace_scotland", None)
    client_args.pop("canary", None)
    client_args["dataset_path"] = "datasets/acn"
    client_args["seed"] = args.seed
    client_args["dp_mode"] = args.dp_mode
    client_args["epsilon"] = args.epsilon

    server_args = server_cfg["components"][0]["args"]
    assert server_cfg["components"][0]["id"] == "aggregator", "components[0] non e' l'aggregator?"
    server_args["seed"] = args.seed
    server_args["dp_mode"] = args.dp_mode
    server_args["epsilon"] = args.epsilon

    with open(CLIENT_PATH, "w") as f:
        json.dump(client_cfg, f, indent=2)
        f.write("\n")
    with open(SERVER_PATH, "w") as f:
        json.dump(server_cfg, f, indent=2)
        f.write("\n")

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    # Fix 2026-09-14 (bug reale trovato prima di lanciare lo sweep epsilon
    # 0.5/0.1): il nome non includeva epsilon, quindi due run con stesso
    # seed+dp_mode ma epsilon diverso (es. central/seed42/eps=1.0 poi
    # central/seed42/eps=0.5) si sovrascrivevano a vicenda in silenzio —
    # esattamente la stessa classe di bug (snapshot sbagliato usato per la
    # rianalisi) che questo script era nato per evitare. Danno reale di
    # questo giro: lo snapshot storico central/seed42/eps=1.0 e' stato perso
    # (nessun impatto: quell'analisi, task #94, era gia' completa e salvata
    # nel suo JSON risultati, che non dipende da questo file per esistere).
    snapshot_path = (
        SNAPSHOT_DIR
        / f"config_fed_client_seed{args.seed}_{args.dp_mode}_eps{args.epsilon}.json"
    )
    with open(snapshot_path, "w") as f:
        json.dump(client_cfg, f, indent=2)
        f.write("\n")

    print(f"OK — dp_mode={args.dp_mode} seed={args.seed} epsilon={args.epsilon}")
    print(f"  {CLIENT_PATH}")
    print(f"  {SERVER_PATH}")
    print(f"  snapshot: {snapshot_path}")
    print("\nProssimo passo (manuale, console admin NVFLARE):")
    print("  bash /workspace/startup/fl_admin.sh")
    print("  submit_job /workspace/jobs/chargeshield_poc")


if __name__ == "__main__":
    main()
