# Il limite teorico della DP, e perché ai nostri ε è vacuo

Stato: documento OPERATIVO. Aggiornato al 2026-09-21.
Codice: `scripts/compute_pes.py::humphries_bound`.
Tabella: foglio `Utility_privacy_limite` in `risultati/Matrice_sintesi.xlsx`.

---

## 1. Che cos'è il limite

Una garanzia $(\varepsilon,\delta)$-DP pone un tetto a quanto bene *qualunque*
avversario possa distinguere fra due dataset adiacenti. Tradotto in termini di
membership inference, il vantaggio massimo ottenibile è

$$\mathrm{Adv}_{\max} = \frac{e^{\varepsilon} - 1 + 2\delta}{e^{\varepsilon} + 1}$$

(Humphries et al.; è una versione più stretta del bound di Yeom perché tiene
conto di $\delta$.) Poiché $\mathrm{AUC} = 0.5 + \mathrm{Adv}/2$, il tetto
sull'AUC è $0.5 + \mathrm{Adv}_{\max}/2$.

---

## 2. Quale ε va messo nella formula

**Quello cumulativo, non quello per round.** Il bound deve coprire il
*transcript* che l'avversario osserva davvero. Il nostro avversario (Scenario 1)
vede un update per round per dieci round: la garanzia che lo riguarda è quella
composta, non quella di una singola release.

Con composizione base, $\varepsilon_{\text{tot}} = T \cdot \varepsilon$. A
$T = 10$:

| ε per round | ε_tot | Adv massima | **AUC massima teorica** | AUC misurata |
|---|---|---|---|---|
| 0.1 | 1.0 | 0.4621 | **0.7311** | ~0.50 |
| 0.5 | 5.0 | 0.9866 | **0.9933** | ~0.50 |
| 1.0 | 10.0 | 0.99991 | **0.99996** | ~0.50 |

Per riferimento, il bound sull'ε **per round** (che non è quello pertinente):
0.5250 a ε=0.1, 0.6225 a ε=0.5, 0.7311 a ε=1.0.

---

## 3. Il punto: il bound è vacuo

A $\varepsilon_{\text{tot}} = 10$ la garanzia permette un'AUC di **0.99996**.
Un attacco quasi perfetto sarebbe pienamente compatibile con la garanzia
formale che stiamo dichiarando.

Ne segue che **la frase "i nostri attacchi stanno sotto il limite teorico" non
è informativa**. Non è falsa: è vera e priva di contenuto, perché il limite non
esclude quasi nulla. Scriverla come se fosse una conferma di sicurezza sarebbe
fuorviante, ed è un errore che un revisore riconosce subito.

Va usata al contrario. La distanza fra ciò che il bound permette (0.99996) e
ciò che misuriamo (0.50) **quantifica quanto la garanzia formale sia lasca
rispetto al comportamento reale del sistema**. Quello sì è un contributo, ed è
coerente con Jayaraman & Evans: le garanzie DP nella pratica sono
sistematicamente molto più permissive di ciò che gli attacchi noti riescono a
sfruttare.

**Formulazione da usare nei documenti e nel paper:**

> Alla composizione base su dieci round, il bound teorico permette un'AUC fino a
> 0.99996 a ε per round pari a 1.0. Riportiamo il confronto non come conferma di
> sicurezza — a quel livello il bound non esclude quasi nulla — ma come misura
> della distanza fra la garanzia dichiarata e il comportamento osservato.

---

## 4. Tre avvertenze sull'uso del bound in questo progetto

**Il nostro ε non è una garanzia verificata.** Non implementiamo DP-SGD nel
braccio client-level: clipping e rumore agiscono una volta per round
sull'update aggregato del client, con 50 epoche locali dentro il round e nessun
clipping per-campione. L'ε per round è un'etichetta di configurazione che
calibra σ, non una garanzia dimostrata. Comporre un'etichetta produce
un'etichetta, e il bound calcolato su quella eredita lo stesso status.

**L'adiacenza è quella sbagliata rispetto all'attacco.** Il meccanismo è
client-level; gli attacchi misurano il record. Il bound sopra è sul client, gli
AUC misurati sono sul record. Vanno messi nella stessa tabella per confronto
visivo, ma non sono la stessa quantità, e va detto.

**La composizione avanzata non aiuta qui.** Al nostro punto operativo
(ε=1.0, δ=1e-5, T=10) la composizione avanzata dà un ε peggiore di quella base,
quindi il bound base è quello che vale. Vedi `sections/preliminaries.tex` e
`scripts/compute_advanced_composition.py`.

---

## 5. Riprodurre la tabella

```bash
python3 - <<'EOF'
import math
for e in (0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
    for T, eti in ((1, e), (10, 10*e)):
        a = (math.exp(eti) - 1) / (math.exp(eti) + 1)
        print(f"eps={e:5.1f} T={T:2d} eps_tot={eti:6.1f} Adv_max={a:.6f} AUC_max={0.5+a/2:.6f}")
EOF
```

Il foglio `Utility_privacy_limite` di `risultati/Matrice_sintesi.xlsx` riporta,
sulla stessa riga di ogni cella sperimentale, costo, privacy misurata e limite
teorico, così che il confronto sia leggibile senza incrociare tabelle.
