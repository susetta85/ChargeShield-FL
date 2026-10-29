# Campagna v3: verifica delle previsioni scritte prima del lancio

### V18 - riproducibilita' (stessa macchina, stesso seed)
- Coppie v3/v2 confrontate: 171 (39 celle). Loss sull'holdout identica: 171/171; AUC di Yeom identica: 171/171; massima differenza relativa della loss 0.00e+00.
- Esito: confermata.

### V19 - ACN senza DP, attacchi realistici
- v3_acn_nodp_mu0: 30/30 valori (6 forme x 5 seed) in [0,45; 0,56]; intervallo 0.497-0.512.
  - LiRA on realistico <= universo + 0,01: 5/5 seed (si, soglia 4).
  - RMIA on realistico <= universo + 0,01: 5/5 seed (si, soglia 4).
- v3_acn_nodp_mu0.01: 30/30 valori (6 forme x 5 seed) in [0,45; 0,56]; intervallo 0.487-0.509.
  - LiRA on realistico <= universo + 0,01: 5/5 seed (si, soglia 4).
  - RMIA on realistico <= universo + 0,01: 5/5 seed (si, soglia 4).

### V20 - con DP l'AUC realistica non supera quella senza DP (+0,01)
- Confronti (cella con DP, forma, seed): 630; sopra +0,01: 57 (9%).
- Differenze massime: acn_subject_s2 LiRA off (aus.) +0.025; acn_eps4 LiRA on +0.021; acn_subject_s5 LiRA off (aus.) +0.020; acn_subject_s2 RMIA off (aus.) +0.020; acn_subject_s2 RMIA off (OUT) +0.020.
- Lettura: senza DP l'AUC e' gia' circa 0,50, quindi le differenze con DP sono rumore di seed; una quota di superamenti di 0,01 e' attesa anche senza alcun effetto della DP.

### V21 - shadow con dati ausiliari, celle senza DP
- acn4f_nodp_mu0: FPR realizzato 14.60%-47.15%, fuori intervallo 5/5 (si); |vantaggio| max 0.012 (si).
- acn4f_nodp_mu0.01: FPR realizzato 1.56%-10.99%, fuori intervallo 2/5 (no); |vantaggio| max 0.007 (si).
- acn_nodp_mu0: FPR realizzato 0.49%-2.98%, fuori intervallo 0/5 (no); |vantaggio| max 0.009 (si).
- acn_nodp_mu0.01: FPR realizzato 1.50%-2.55%, fuori intervallo 0/5 (no); |vantaggio| max 0.005 (si).
- canaryA_nodp: FPR realizzato 0.15%-0.15%, fuori intervallo 1/1 (no); |vantaggio| max 0.002 (si).
- canaryB_nodp: FPR realizzato 0.17%-0.17%, fuori intervallo 1/1 (no); |vantaggio| max 0.002 (si).
- sc3_nodp_mu0: FPR realizzato 0.20%-15.86%, fuori intervallo 5/5 (si); |vantaggio| max 0.008 (si).
- sc3_nodp_mu0.01: FPR realizzato 0.31%-14.89%, fuori intervallo 1/5 (no); |vantaggio| max 0.006 (si).

### V22 - classificatore di tipo Shokri contro soglia sulla loss, celle senza DP
- acn4f_nodp_mu0: |clf - soglia| <= 0,01 in 4/5; clf sopra di oltre 0,01 in 0/5 (segnale se >= 3).
- acn4f_nodp_mu0.01: |clf - soglia| <= 0,01 in 4/5; clf sopra di oltre 0,01 in 1/5 (segnale se >= 3).
- acn_nodp_mu0: |clf - soglia| <= 0,01 in 2/5; clf sopra di oltre 0,01 in 2/5 (segnale se >= 3).
- acn_nodp_mu0.01: |clf - soglia| <= 0,01 in 4/5; clf sopra di oltre 0,01 in 1/5 (segnale se >= 3).
- canaryA_nodp: |clf - soglia| <= 0,01 in 1/1; clf sopra di oltre 0,01 in 0/1 (segnale se >= 3).
- canaryB_nodp: |clf - soglia| <= 0,01 in 1/1; clf sopra di oltre 0,01 in 0/1 (segnale se >= 3).
- sc3_nodp_mu0: |clf - soglia| <= 0,01 in 4/5; clf sopra di oltre 0,01 in 1/5 (segnale se >= 3).
- sc3_nodp_mu0.01: |clf - soglia| <= 0,01 in 3/5; clf sopra di oltre 0,01 in 0/5 (segnale se >= 3).

### V23 - avversario partecipante (solo modelli globali)
- partecipante_acn_nodp_mu0.01: LiRA e RMIA online 0.492-0.504 (in [0,45; 0,56]: si); non sopra il server + 0,01 in 5/5 seed (si).
- partecipante_acn_eps64_mu0.01: LiRA e RMIA online 0.496-0.505 (in [0,45; 0,56]: si); non sopra il server + 0,01 in 4/5 seed (si).
- partecipante_acn_central_eps64: LiRA e RMIA online 0.492-0.506 (in [0,45; 0,56]: si); non sopra il server + 0,01 in 4/5 seed (si).

### V25 - ablazione solo taglio / solo rumore, ACN a sei feature
- Solo taglio, FedAvg con loss piu' alta di FedProx: 5/5 seed (si); rapporto FedAvg/FedProx 2.11-6.68.
- Solo rumore contro DP completa (FedProx): rapporto 0.21-6.15, entro un fattore 1,5 in 1/5 seed (no).

### V26 - 30 round: i round 1-10 coincidono con la cella a 10 round
- v3_30r_mu0_nodp contro v3_acn_nodp_mu0: seed con round 1-10 identici 5/5; massima differenza relativa 0.00e+00.
- v3_30r_mu0.01_nodp contro v3_acn_nodp_mu0.01: seed con round 1-10 identici 5/5; massima differenza relativa 0.00e+00.
- v3_30r_mu0_eps64 contro v3_acn_eps64_mu0: seed con round 1-10 identici 5/5; massima differenza relativa 0.00e+00.
- v3_30r_mu0.01_eps64 contro v3_acn_eps64_mu0.01: seed con round 1-10 identici 5/5; massima differenza relativa 0.00e+00.
- v3_sc3_30r_mu0_nodp contro v3_sc3_nodp_mu0: seed con round 1-10 identici 5/5; massima differenza relativa 0.00e+00.
- v3_sc3_30r_mu0.01_nodp contro v3_sc3_nodp_mu0.01: seed con round 1-10 identici 5/5; massima differenza relativa 0.00e+00.
- v3_sc3_30r_mu0_eps64 contro v3_sc3_eps64_mu0: seed con round 1-10 identici 5/5; massima differenza relativa 0.00e+00.
- v3_sc3_30r_mu0.01_eps64 contro v3_sc3_eps64_mu0.01: seed con round 1-10 identici 5/5; massima differenza relativa 0.00e+00.
- Differenza FedAvg - FedProx al round 30 (loss sull'holdout, media sui seed, senza previsione):
  - ACN nodp: FedAvg/FedProx 0.56 (min 0.46, max 0.78, FedAvg peggiore in 0/10 seed).
  - ACN eps64: FedAvg/FedProx 1.56 (min 0.39, max 3.55, FedAvg peggiore in 7/10 seed).
  - Scozia nodp: FedAvg/FedProx 0.17 (min 0.05, max 0.45, FedAvg peggiore in 0/10 seed).
  - Scozia eps64: FedAvg/FedProx 1.52 (min 0.43, max 3.33, FedAvg peggiore in 6/10 seed).


### V24 - canary per sessione senza DP: loss grezza sopra 0,5 in ogni braccio, A + B sopra 1 in ogni seed
- seed 456 (Mac di Domenico): A 0.557, B 0.578, somma 1.135 (si).
- Seed raccolti: 456. La previsione e' su 3 seed (42, 123, 456); 789 e 1234 sono un'estensione decisa il 23 ottobre.

### V27 - entita', senza DP: AUC primaria in [0,45; 0,58]; DP per soggetto non sopra +0,02
- ACN: 5/5 seed in [0,45; 0,58] (AUC 0.490-0.566).
  - subject_s2: non sopra senza DP + 0,02 in 2/3 seed.
  - subject_s5: non sopra senza DP + 0,02 in 5/5 seed.
- Scozia: 5/5 seed in [0,45; 0,58] (AUC 0.495-0.561).
  - sc3subject_s5: non sopra senza DP + 0,02 in 5/5 seed.

### V28 - gruppo, DP per record: segnale se AUC primaria > 0,55 in almeno 2 seed su 3 con sigma 2
- ACN record_s2: AUC primaria 0.492-0.517, sopra 0,55 in 0/3 seed.
- ACN record_s5: AUC primaria 0.489-0.515, sopra 0,55 in 0/3 seed.
- Scozia sc3record_s1: AUC primaria 0.496-0.521, sopra 0,55 in 0/5 seed.
