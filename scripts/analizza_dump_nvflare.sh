#!/usr/bin/env bash
# Generato 2026-09-21 — analisi dei 25 dump NVFLARE con provenienza certa.
# I 3 dump incoerenti (ri-esecuzioni senza nuovo snapshot) sono esclusi:
# vanno decisi a mano, vedi experiments/_dump_nvflare_provenienza.csv.
set -e
cd "$(dirname "$0")/.."

echo "[1/25] local eps=0.5 seed=42"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_214538_938e70.pkl \
    --seed 42 --n-shadow 16 \
    --sweep-dir experiments/nvflare-local-eps0.5-seed42

echo "[2/25] local eps=0.1 seed=42"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_220011_895da8.pkl \
    --seed 42 --n-shadow 16 \
    --sweep-dir experiments/nvflare-local-eps0.1-seed42

echo "[3/25] local eps=0.5 seed=123"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_220944_8f4078.pkl \
    --seed 123 --n-shadow 16 \
    --sweep-dir experiments/nvflare-local-eps0.5-seed123

echo "[4/25] local eps=0.1 seed=123"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_221820_e48bfc.pkl \
    --seed 123 --n-shadow 16 \
    --sweep-dir experiments/nvflare-local-eps0.1-seed123

echo "[5/25] local eps=0.1 seed=456"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_223011_37cc07.pkl \
    --seed 456 --n-shadow 16 \
    --sweep-dir experiments/nvflare-local-eps0.1-seed456

echo "[6/25] local eps=0.5 seed=456"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_223825_85475e.pkl \
    --seed 456 --n-shadow 16 \
    --sweep-dir experiments/nvflare-local-eps0.5-seed456

echo "[7/25] local eps=0.5 seed=789"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_225419_7bc7f5.pkl \
    --seed 789 --n-shadow 16 \
    --sweep-dir experiments/nvflare-local-eps0.5-seed789

echo "[8/25] local eps=0.1 seed=789"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_230207_ba90af.pkl \
    --seed 789 --n-shadow 16 \
    --sweep-dir experiments/nvflare-local-eps0.1-seed789

echo "[9/25] local eps=0.5 seed=1234"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_231936_17849c.pkl \
    --seed 1234 --n-shadow 16 \
    --sweep-dir experiments/nvflare-local-eps0.5-seed1234

echo "[10/25] local eps=0.1 seed=1234"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_232851_e6a4d0.pkl \
    --seed 1234 --n-shadow 16 \
    --sweep-dir experiments/nvflare-local-eps0.1-seed1234

echo "[11/25] dp-fedavg eps=1.0 seed=42"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_233649_b6a60e.pkl \
    --seed 42 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps1.0-seed42

echo "[12/25] dp-fedavg eps=0.5 seed=42"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260917_234906_be8531.pkl \
    --seed 42 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps0.5-seed42

echo "[13/25] dp-fedavg eps=0.1 seed=42"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260918_000607_773626.pkl \
    --seed 42 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps0.1-seed42

echo "[14/25] dp-fedavg eps=1.0 seed=123"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260918_001415_d14494.pkl \
    --seed 123 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps1.0-seed123

echo "[15/25] dp-fedavg eps=0.5 seed=123"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260918_002416_109703.pkl \
    --seed 123 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps0.5-seed123

echo "[16/25] dp-fedavg eps=0.1 seed=123"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260918_003148_e78552.pkl \
    --seed 123 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps0.1-seed123

echo "[17/25] dp-fedavg eps=1.0 seed=456"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260919_114228_15c819.pkl \
    --seed 456 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps1.0-seed456

echo "[18/25] dp-fedavg eps=0.5 seed=456"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260919_115027_dfd042.pkl \
    --seed 456 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps0.5-seed456

echo "[19/25] dp-fedavg eps=0.1 seed=456"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260919_132026_a6079c.pkl \
    --seed 456 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps0.1-seed456

echo "[20/25] dp-fedavg eps=1.0 seed=789"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260919_174900_d65dc2.pkl \
    --seed 789 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps1.0-seed789

echo "[21/25] dp-fedavg eps=0.5 seed=789"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260919_201212_6d54b5.pkl \
    --seed 789 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps0.5-seed789

echo "[22/25] dp-fedavg eps=0.1 seed=789"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260919_202233_182adf.pkl \
    --seed 789 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps0.1-seed789

echo "[23/25] dp-fedavg eps=1.0 seed=1234"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260919_203109_f41e2e.pkl \
    --seed 1234 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps1.0-seed1234

echo "[24/25] dp-fedavg eps=0.5 seed=1234"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260919_203927_987424.pkl \
    --seed 1234 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps0.5-seed1234

echo "[25/25] dp-fedavg eps=0.1 seed=1234"
python3 scripts/run_nvflare_mia.py \
    --fl-results experiments/nvflare_fl_results_20260919_204916_2c69ef.pkl \
    --seed 1234 --n-shadow 16 \
    --sweep-dir experiments/nvflare-dp-fedavg-eps0.1-seed1234

