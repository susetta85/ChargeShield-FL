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

Uso (baseline ACN, invariato):
    python3 scripts/set_nvflare_seed.py --dp-mode central --seed 123
    python3 scripts/set_nvflare_seed.py --dp-mode local --seed 42

Uso (canary positive control, Sprint 10zz+105 — richiede il fix di tagging
gia' applicato a chargeshield_executor.py::_inject_canary_members() e a
run_nvflare_mia.py, altrimenti si riproduce silenziosamente lo stesso bug
di shadow-contamination gia' diagnosticato in simulazione):
    python3 scripts/set_nvflare_seed.py --dp-mode central --seed 123 \
        --canary-enabled --canary-site office1 \
        --canary-n-templates 5 --canary-n-duplicates 30

Uso (ChargePlace Scotland, task #37):
    python3 scripts/set_nvflare_seed.py --dp-mode central --seed 123 \
        --dataset-adapter chargeplace_scotland \
        --scotland-metadata-dir datasets/chargeplace_scotland/metadata \
        --scotland-session-files datasets/chargeplace_scotland/sessions_2019.json \
        --scotland-site-mapping caltech=Glasgow_City

Dopo aver lanciato questo script, il passo successivo resta manuale:
    bash /workspace/startup/fl_admin.sh   # login admin@chargeshield.local
    submit_job /workspace/jobs/chargeshield_poc
"""

import argparse
import json
from pathlib import Path
from typing import Any

CONFIG_DIR = Path("nvflare/jobs/chargeshield_poc/app/config")
CLIENT_PATH = CONFIG_DIR / "config_fed_client.json"
SERVER_PATH = CONFIG_DIR / "config_fed_server.json"
SNAPSHOT_DIR = CONFIG_DIR / "seed_snapshots"

VALID_MODES = ("dp-fedavg", "central", "local")
VALID_ADAPTERS = ("acn", "chargeplace_scotland")


def _parse_site_mapping(pairs: list[str]) -> dict[str, str]:
    """'caltech=Glasgow_City' 'jpl=Edinburgh' -> {'caltech': 'Glasgow_City', ...}."""
    mapping: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(
                f"--scotland-site-mapping: '{pair}' non e' nel formato site=local_authority"
            )
        site, authority = pair.split("=", 1)
        mapping[site.strip()] = authority.strip()
    return mapping


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dp-mode", required=True, choices=VALID_MODES)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--epsilon", type=float, default=1.0)

    # Canary positive control (Sprint 10zz+105). Tutti opt-in: default
    # invariato = nessun canary, esattamente come prima di questa estensione.
    p.add_argument("--canary-enabled", action="store_true")
    p.add_argument("--canary-site", default="office1")
    p.add_argument("--canary-n-templates", type=int, default=5)
    p.add_argument("--canary-n-duplicates", type=int, default=30)
    p.add_argument(
        "--canary-n-nonmember-templates",
        type=int,
        default=None,
        help="Default: uguale a --canary-n-templates (letto solo da "
        "scripts/run_nvflare_mia.py in fase di rianalisi offline — "
        "l'executor live inietta solo i member, mai i nonmember).",
    )

    # ChargePlace Scotland (task #37). Default invariato = adapter 'acn'.
    p.add_argument("--dataset-adapter", choices=VALID_ADAPTERS, default="acn")
    p.add_argument("--scotland-metadata-dir", default=None)
    p.add_argument("--scotland-session-files", nargs="*", default=None)
    p.add_argument(
        "--scotland-site-mapping",
        nargs="*",
        default=None,
        help=(
            "Coppie site=local_authority. Il valore deve corrispondere "
            "ESATTAMENTE al campo local_authority dei metadati "
            "(CPID_and_local_authority.xlsx), che contiene spazi e non "
            "underscore: usare le virgolette, es. \"caltech=Glasgow City\". "
            "L'esempio precedente in questo help (caltech=Glasgow_City) era "
            "sbagliato: _parse_site_mapping() non converte gli underscore, "
            "quindi 'Glasgow_City' non corrisponde a nessuna delle 32 "
            "council area e il filtro restituisce zero sessioni in silenzio "
            "(corretto 2026-09-16, Sprint 10zz+114)."
        ),
    )
    args = p.parse_args()

    if args.dataset_adapter == "chargeplace_scotland" and not args.scotland_metadata_dir:
        raise SystemExit(
            "--dataset-adapter chargeplace_scotland richiede --scotland-metadata-dir"
        )

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
    # Reset SEMPRE a stato pulito prima di riapplicare (sotto) solo cio' che
    # e' stato esplicitamente richiesto in questa invocazione — cosi' i nuovi
    # flag canary/Scotland restano opt-in senza perdere la garanzia "run
    # pulito di default" che questo script dava gia'.
    client_args.pop("dataset_adapter", None)
    client_args.pop("chargeplace_scotland", None)
    client_args.pop("canary", None)
    client_args["dataset_path"] = "datasets/acn"
    client_args["seed"] = args.seed
    client_args["dp_mode"] = args.dp_mode
    client_args["epsilon"] = args.epsilon

    if args.canary_enabled:
        canary_cfg: dict[str, Any] = {
            "enabled": True,
            "site": args.canary_site,
            "n_templates": args.canary_n_templates,
            "n_duplicates": args.canary_n_duplicates,
        }
        # n_nonmember_templates e' usato SOLO da run_nvflare_mia.py (il suo
        # default e' gia' n_templates se la chiave manca) — la includiamo
        # esplicitamente solo se l'utente l'ha specificata diversa, per non
        # sporcare il config con una chiave l'executor live ignora comunque.
        if args.canary_n_nonmember_templates is not None:
            canary_cfg["n_nonmember_templates"] = args.canary_n_nonmember_templates
        client_args["canary"] = canary_cfg

    if args.dataset_adapter == "chargeplace_scotland":
        client_args["dataset_adapter"] = "chargeplace_scotland"
        client_args["chargeplace_scotland"] = {
            "metadata_dir": args.scotland_metadata_dir,
            "session_files": args.scotland_session_files or [],
            "site_mapping": _parse_site_mapping(args.scotland_site_mapping or []),
        }

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
    # Suffisso canary/scotland nel nome snapshot (Sprint 10zz+105) — stessa
    # motivazione del fix epsilon del 2026-09-14: senza distinguere questi
    # run nel filename, uno snapshot canary/Scotland sovrascriverebbe in
    # silenzio lo snapshot ACN "pulito" con stesso seed/dp_mode/epsilon (o
    # viceversa), riproponendo la classe di bug che questo script esiste per
    # evitare (snapshot sbagliato usato dalla rianalisi offline).
    suffix = ""
    if args.canary_enabled:
        suffix += "_canary"
    if args.dataset_adapter == "chargeplace_scotland":
        suffix += "_scotland"
    snapshot_path = (
        SNAPSHOT_DIR
        / f"config_fed_client_seed{args.seed}_{args.dp_mode}_eps{args.epsilon}{suffix}.json"
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
