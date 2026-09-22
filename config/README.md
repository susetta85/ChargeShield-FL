# Indice dei config

> **Stato: documento OPERATIVO.** Una riga per file, aggiornato il 2026-09-22. Ogni
> config è una cella: prima di crearne uno nuovo cercare qui se esiste già. La base
> di tutti è `experiment.yaml`; le colonne dicono cosa cambia rispetto a quella base
> (3 siti, 10 round, 50 epoche, 6 feature, ε = 1, n_shadow 16, FedProx mu 0.01).
> `--epsilon`, `--rounds`, `--epochs`, `--seed`, `--dp-mode`, `--no-dp` da riga di
> comando hanno la precedenza sul file.

## Regime naturale: RQ1, RQ2, RQ3 (`docs/ESPERIMENTI.md`)

| file | cosa cambia | uso |
|---|---|---|
| `experiment.yaml` | niente: è la base | no-DP con `--no-dp`; client-level con `--dp-mode` ed `--epsilon` |
| `experiment_rq1_eps{2,4,8,16}.yaml` | solo ε | E-A, sweep di ε |
| `experiment_rq1_recorddp_nm{0.5,1,2,5}.yaml` | `record_dp` attivo con quel `noise_multiplier` | E-B, **sempre con `--no-dp`** |
| `experiment_rq2_per_site.yaml`, `experiment_rq2_iid.yaml` | solo `partition.strategy` | E-D, RQ2 |
| `experiment_robustness_entity_split.yaml` | `split.strategy: entity_aware` | robustezza dello split, già eseguito a 5 seed |
| `experiment_chargeplace_scotland.yaml` | secondo dataset, 3 council area | futuro; una sola run di prova |

## Diagnostiche LiRA: non usare finché lo scorer è congelato

| file | cosa cambia |
|---|---|
| `experiment_floor_independent.yaml` | `lira.floor_mode: independent` |
| `experiment_matched_formula_scoring.yaml` | `lira.member_scoring: matched_formula` |
| `experiment_shadow_cold.yaml` | `lira.shadow_init: cold` |
| `experiment_hour_circular_encoding.yaml` | ora codificata in seno e coseno, Office 1, 3 round |

## Canary bilanciato: il protocollo che regge (`docs/CanaryPositiveControl.md`)

Tutti su Office 1, 3 round, 7 feature, capacità (32, 16)/8, k = 20 template membro e
20 non membro, 30 duplicati, `paired_split`, n_shadow 8, shadow cold. Lanciare
sempre la coppia base e swap con lo stesso seed.

| file | cosa cambia |
|---|---|
| `experiment_canary_balanced.yaml`, `_swap.yaml` | il riferimento, 1000 epoche, BatchNorm |
| `experiment_canary_e1000_group.yaml`, `_swap.yaml` | come sopra con GroupNorm, confrontabile con le celle record-DP |
| `experiment_canary_balanced_e100_group.yaml`, `experiment_canary_e100_group_swap.yaml` | 100 epoche, GroupNorm |
| `experiment_canary_e10_group.yaml` | 10 epoche, GroupNorm |
| `experiment_canary_balanced_nshadow32.yaml`, `_swap.yaml` | n_shadow 32, ablation |
| `experiment_canary_balanced_indepfloor.yaml` | floor indipendente, ablation |

## Canary con DP record-level (regime canary, `--no-dp` per il client-level)

| file | `noise_multiplier` | epoche | nota |
|---|---|---|---|
| `experiment_rdp_e1000_s5.yaml`, `_swap.yaml` | 5 | 1000 | la cella ε dichiarato 7.15, un seed; ε da ricalcolare (segnalazione 4) |
| `experiment_rdp_e1000_s10.yaml` | 10 | 1000 | |
| `experiment_rdp_auditor_tuned.yaml` | 5 | 1000 | come s5 più `auditor_tuned.yaml` |
| `experiment_canary_balanced_recorddp.yaml` | 1 | 1000 | |
| `experiment_canary_balanced_recorddp_fast.yaml` | 1 | 100 | |
| `experiment_canary_recorddp_nm01.yaml` | 0.1 | 100 | |
| `experiment_canary_recorddp_nm005.yaml`, `_swap.yaml` | 0.05 | 100 | |
| `experiment_rdp_eps163.yaml`, `experiment_rdp_eps474.yaml` | 2, 1 | 10 | |
| `experiment_canary_balanced_recorddp_nm5.yaml` | 5 | 5 | smoke test, non un esperimento |

## Canary superati: solo il braccio base è un controllo positivo, lo swap non è un controllo negativo

Gruppi sbilanciati, 5 template membro e 20 non membro: con questa forma lo scambio
dei ruoli estrae insiemi disgiunti e non è uno scambio. Restano come storia.

| file | sito | duplicati | nota |
|---|---|---|---|
| `experiment_canary_positive_control.yaml` | Office 1 | 30 | 6 feature, capacità storica |
| `experiment_canary_positive_control_caltech.yaml` | Caltech | 30 | segnale non riprodotto |
| `experiment_canary_positive_control_caltech_highdensity.yaml` | Caltech | 30 su 84 template | densità aggregata pari a Office 1, amplificazione per record no |
| `experiment_canary_positive_control_caltech_amplification_matched.yaml` | Caltech | 561 | amplificazione per record pari a Office 1 |
| `experiment_canary_positive_control_jpl.yaml` | JPL | 30 | segnale non riprodotto |
| `experiment_canary_positive_control_chargeplace_scotland.yaml` | Glasgow City | 4650, 50 epoche | mai eseguito |
| `experiment_canary_coldstart.yaml`, `_swap.yaml` | Office 1 | 30 | 6 feature, shadow cold |
| `experiment_canary_maxmemo.yaml`, `_swap.yaml` | Office 1 | 30 | 7 feature, capacità (32, 16)/8 |
| `experiment_canary_swap_control.yaml` | Office 1 | 30 | swap sbilanciato |

## Calibrazione della memorizzazione naturale: chiusa, cinque assi tutti nulli

`experiment_overfit_calibration.yaml` (Office 1, 50 epoche), `_bigcap` (capacità
3.3x, 1000 epoche), `_richfeat` (7 feature), `_combined` (capacità e feature),
`_altsite` (Caltech 2021). Servivano a stabilire se il modello può memorizzare
senza canary: no.

## Auditor

`auditor.yaml`: parametri del Privacy Auditor con `total_rounds_budget: 1000`, per
cui le soglie di budget non scattano mai in 10 round. `auditor_tuned.yaml`: budget
su 3 round, selezionabile con `CHARGESHIELD_AUDITOR_CONFIG`. Il campo `attacks` è
storico e non seleziona nulla.

## Infrastruttura e legacy

`flare.yaml` resta in questa cartella solo perché lo scaffolding non collegato
`src/flare/flare_connector.py` e il suo test lo cercano qui: i suoi valori non sono
quelli degli esperimenti. `_legacy_unused/` contiene i config del progetto iniziale a
12 nodi e 4 cluster mai costruito (`clusters.yaml`, `nodes.yaml`, `nodes/`,
`protocols.yaml`, `framework.yaml`, `datasets.yaml`) e il vecchio
`experiment_overfit_control.yaml`, che è solo commenti. Non usarli.
