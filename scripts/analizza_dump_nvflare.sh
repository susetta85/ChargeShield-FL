#!/usr/bin/env bash
# Rigenerato 2026-09-22 dopo il bug dello split seed (revisione A.1).
# --client-config e' OBBLIGATORIO: senza, run_nvflare_mia.py legge il seed
# dello split dal config live (1234) e ricostruisce una partizione diversa
# da quella dei client, con AUC che tende a 0.5 per costruzione.
set -e
cd "$(dirname "$0")/.."

echo "[1/25] local eps=0.5 seed=42"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_214538_938e70.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed42_local_eps0.5.json \
    --seed 42 --n-shadow 16 \
    --sweep-dir experiments/nvflare-local-eps0.5-seed42

echo "[2/25] local eps=0.1 seed=42"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_220011_895da8.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed42_local_eps0.1.json \
    --seed 42 --n-shadow 16 \
    --sweep-dir experiments/nvflare-local-eps0.1-seed42

echo "[3/25] local eps=0.5 seed=123"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_220944_8f4078.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed123_local_eps0.5.json \
    --seed 123 --n-shadow 16 \
    --sweep-dir experiments/nvflare-local-eps0.5-seed123

echo "[4/25] local eps=0.1 seed=123"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_221820_e48bfc.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed123_local_eps0.1.json \
    --seed 123 --n-shadow 16 \
    --sweep-dir experiments/nvflare-local-eps0.1-seed123

echo "[5/25] local eps=0.1 seed=456"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_223011_37cc07.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed456_local_eps0.1.json \
    --seed 456 --n-shadow 16 \
    --sweep-dir experiments/nvflare-local-eps0.1-seed456

echo "[6/25] local eps=0.5 seed=456"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_223825_85475e.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed456_local_eps0.5.json \
    --seed 456 --n-shadow 16 \
    --sweep-dir experiments/nvflare-local-eps0.5-seed456

echo "[7/25] local eps=0.5 seed=789"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_225419_7bc7f5.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed789_local_eps0.5.json \
    --seed 789 --n-shadow 16 \
    --sweep-dir experiments/nvflare-local-eps0.5-seed789

echo "[8/25] local eps=0.1 seed=789"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_230207_ba90af.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed789_local_eps0.1.json \
    --seed 789 --n-shadow 16 \
    --sweep-dir experiments/nvflare-local-eps0.1-seed789

echo "[9/25] local eps=0.5 seed=1234"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_231936_17849c.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed1234_local_eps0.5.json \
    --seed 1234 --n-shadow 16 \
    --sweep-dir experiments/nvflare-local-eps0.5-seed1234

echo "[10/25] local eps=0.1 seed=1234"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_232851_e6a4d0.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed1234_local_eps0.1.json \
    --seed 1234 --n-shadow 16 \
    --sweep-dir experiments/nvflare-local-eps0.1-seed1234

echo "[11/25] dp-fedavg eps=1.0 seed=42"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_233649_b6a60e.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed42_dp-fedavg_eps1.0.json \
    --seed 42 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps1.0-seed42

echo "[12/25] dp-fedavg eps=0.5 seed=42"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_234906_be8531.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed42_dp-fedavg_eps0.5.json \
    --seed 42 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps0.5-seed42

echo "[13/25] dp-fedavg eps=0.1 seed=42"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260918_000607_773626.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed42_dp-fedavg_eps0.1.json \
    --seed 42 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps0.1-seed42

echo "[14/25] dp-fedavg eps=1.0 seed=123"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260918_001415_d14494.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed123_dp-fedavg_eps1.0.json \
    --seed 123 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps1.0-seed123

echo "[15/25] dp-fedavg eps=0.5 seed=123"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260918_002416_109703.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed123_dp-fedavg_eps0.5.json \
    --seed 123 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps0.5-seed123

echo "[16/25] dp-fedavg eps=0.1 seed=123"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260918_003148_e78552.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed123_dp-fedavg_eps0.1.json \
    --seed 123 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps0.1-seed123

echo "[17/25] dp-fedavg eps=1.0 seed=456"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260919_114228_15c819.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed456_dp-fedavg_eps1.0.json \
    --seed 456 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps1.0-seed456

echo "[18/25] dp-fedavg eps=0.5 seed=456"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260919_115027_dfd042.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed456_dp-fedavg_eps0.5.json \
    --seed 456 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps0.5-seed456

echo "[19/25] dp-fedavg eps=0.1 seed=456"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260919_132026_a6079c.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed456_dp-fedavg_eps0.1.json \
    --seed 456 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps0.1-seed456

echo "[20/25] dp-fedavg eps=1.0 seed=789"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260919_174900_d65dc2.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed789_dp-fedavg_eps1.0.json \
    --seed 789 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps1.0-seed789

echo "[21/25] dp-fedavg eps=0.5 seed=789"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260919_201212_6d54b5.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed789_dp-fedavg_eps0.5.json \
    --seed 789 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps0.5-seed789

echo "[22/25] dp-fedavg eps=0.1 seed=789"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260919_202233_182adf.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed789_dp-fedavg_eps0.1.json \
    --seed 789 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps0.1-seed789

echo "[23/25] dp-fedavg eps=1.0 seed=1234"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260919_203109_f41e2e.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed1234_dp-fedavg_eps1.0.json \
    --seed 1234 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps1.0-seed1234

echo "[24/25] dp-fedavg eps=0.5 seed=1234"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260919_203927_987424.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed1234_dp-fedavg_eps0.5.json \
    --seed 1234 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps0.5-seed1234

echo "[25/25] dp-fedavg eps=0.1 seed=1234"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260919_204916_2c69ef.pkl \
    --client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed1234_dp-fedavg_eps0.1.json \
    --seed 1234 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps0.1-seed1234

