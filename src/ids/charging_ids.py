# src/ids/charging_ids.py
"""
ChargingIDS — Real Intrusion Detection System per Reti EV in FL
===============================================================
Implementazione concreta di AbstractIDS per ambienti reali di
colonnine di ricarica EV con Federated Learning.

Questo modulo implementa quattro meccanismi di rilevamento reali:

1. CUSUM (Cumulative Sum Control Chart)
   Rileva derive statistiche nel comportamento dei nodi nel tempo.
   A differenza delle soglie fisse, CUSUM si adatta alla baseline
   storica del nodo e rileva cambiamenti graduali.
   Ref: Page, "Continuous Inspection Schemes", Biometrika 1954

2. Krum Byzantine Fault Detection
   Identifica nodi Byzantine confrontando geometricamente i gradienti.
   Un nodo Byzantine ha gradienti molto distanti da tutti gli altri.
   Progettato per tollerare fino a f nodi malicious su n totali.
   Ref: Blanchard et al., "Byzantine Tolerant SGD", NeurIPS 2017

3. Cosine Similarity Analysis
   Rileva model poisoning confrontando la direzione dei gradienti.
   Un nodo che invia gradienti ortogonali (cosine ≈ 0) o opposti
   (cosine < 0) agli altri nodi del cluster è sospetto.

4. FedMIA Integration
   Usa i risultati dell'attacco FedMIA per identificare i nodi
   vulnerabili alla membership inference e quelli che tentano
   di estrarre informazioni sugli altri.

Azioni consigliate:
- MONITOR:  anomalia lieve — monitora i round successivi
- THROTTLE: anomalia media — limita la frequenza di partecipazione
- EXCLUDE:  anomalia grave — escludi dall'aggregazione corrente

Riferimenti aggiuntivi:
- Fung et al., "Mitigating Sybils in FL Poisoning", 2020
- Nasr et al., "Comprehensive Privacy Analysis of Deep Learning", 2019
"""

import math
import yaml
from pathlib import Path
from typing import Any

from src.core.base_ids import AbstractIDS, IDSAlert, RoundAnalysis
from src.core.base_auditor import AuditReport
from src.plugins.attacks.fedmia import FedMIA, MIAResult


def _load_ids_config(config_path: str = "config/auditor.yaml") -> dict:
    """
    Carica la configurazione IDS da auditor.yaml.

    Args:
        config_path: percorso al file di configurazione

    Raises:
        FileNotFoundError: se il file non esiste
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"IDS config not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)["auditor"]


# ─── CUSUM Detector ───────────────────────────────────────────────────────────

class CUSUMDetector:
    """
    CUSUM (Cumulative Sum) detector per deriva statistica.

    Monitora una sequenza di valori per rilevare quando la media
    si sposta significativamente dalla baseline storica.

    Vantaggi rispetto alle soglie fisse:
    - Si adatta alla baseline individuale di ogni nodo
    - Rileva cambiamenti graduali (non solo spike improvvisi)
    - Controllo del false positive rate tramite parametro drift

    Parametri:
    - threshold: soglia CUSUM oltre cui si segnala anomalia (default 5.0)
    - drift:     sensibilità — quanto deviazione tollerare prima di allarmare
                 (default 0.5 — buon compromesso per FL con round variabili)
    """

    def __init__(self, threshold: float = 5.0, drift: float = 0.5):
        """
        Args:
            threshold: soglia di anomalia (più alto = meno sensibile)
            drift:     deviazione minima prima di accumulare il CUSUM
        """
        self._threshold = threshold
        self._drift = drift
        # CUSUM positivo: rileva aumenti sopra la media
        self._cusum_pos: dict[str, float] = {}
        # CUSUM negativo: rileva diminuzioni sotto la media
        self._cusum_neg: dict[str, float] = {}
        # Media storica per ogni nodo
        self._means: dict[str, float] = {}
        # Contatore osservazioni per warm-up
        self._counts: dict[str, int] = {}

    def update(self, node_id: str, value: float) -> bool:
        """
        Aggiorna il CUSUM per un nodo e rileva anomalie.

        Le prime 10 osservazioni servono per il warm-up:
        durante il warm-up la media viene calibrata e non
        vengono emessi alert (troppo pochi dati per decidere).

        Args:
            node_id: identificatore del nodo
            value:   valore osservato nel round corrente

        Returns:
            True se anomalia rilevata, False altrimenti
        """
        if node_id not in self._means:
            # Prima osservazione: inizializza
            self._means[node_id] = value
            self._counts[node_id] = 1
            self._cusum_pos[node_id] = 0.0
            self._cusum_neg[node_id] = 0.0
            return False

        count = self._counts[node_id]

        # Warm-up: prime 10 osservazioni per calibrare la media
        if count < 10:
            self._means[node_id] = (
                self._means[node_id] * count + value
            ) / (count + 1)
            self._counts[node_id] += 1
            return False

        self._counts[node_id] += 1
        mean = self._means[node_id]

        # Aggiorna CUSUM positivo e negativo
        # max(0, ...) azzera quando il CUSUM scende sotto zero
        self._cusum_pos[node_id] = max(
            0.0,
            self._cusum_pos[node_id] + (value - mean) - self._drift,
        )
        self._cusum_neg[node_id] = max(
            0.0,
            self._cusum_neg[node_id] + (mean - value) - self._drift,
        )

        return (
            self._cusum_pos[node_id] > self._threshold or
            self._cusum_neg[node_id] > self._threshold
        )

    def get_cusum_values(self, node_id: str) -> dict[str, float]:
        """
        Restituisce i valori CUSUM correnti per un nodo.
        Utile per logging e analisi nel paper.

        Args:
            node_id: identificatore del nodo

        Returns:
            dizionario con cusum_pos, cusum_neg, mean
        """
        return {
            "cusum_pos": self._cusum_pos.get(node_id, 0.0),
            "cusum_neg": self._cusum_neg.get(node_id, 0.0),
            "mean": self._means.get(node_id, 0.0),
            "count": self._counts.get(node_id, 0),
        }

    def reset(self, node_id: str | None = None) -> None:
        """
        Resetta il CUSUM — per un nodo specifico o tutti.

        Args:
            node_id: se None, resetta tutti i nodi
        """
        if node_id:
            for d in (
                self._cusum_pos, self._cusum_neg,
                self._means, self._counts,
            ):
                d.pop(node_id, None)
        else:
            self._cusum_pos.clear()
            self._cusum_neg.clear()
            self._means.clear()
            self._counts.clear()


# ─── Gradient Analyzer ────────────────────────────────────────────────────────

class GradientAnalyzer:
    """
    Analizza le proprietà geometriche dei gradienti.

    Metodi:
    - flatten:                 appiattisce il gradient dict in lista float
    - l2_norm:                 norma L2 del gradiente
    - cosine_similarity:       similarità coseno tra due gradienti
    - cluster_cosine_analysis: similarità media di ogni nodo vs il cluster
    """

    @staticmethod
    def flatten(gradient: dict[str, Any]) -> list[float]:
        """
        Appiattisce un dizionario gradiente in lista di float.
        I valori non numerici vengono ignorati.

        Args:
            gradient: dizionario layer → pesi

        Returns:
            lista piatta di float
        """
        flat: list[float] = []
        for value in gradient.values():
            if isinstance(value, (int, float)):
                flat.append(float(value))
            elif isinstance(value, list):
                flat.extend(
                    float(v) for v in value
                    if isinstance(v, (int, float))
                )
        return flat

    @staticmethod
    def l2_norm(values: list[float]) -> float:
        """
        Calcola la norma L2 del gradiente.

        Args:
            values: lista di float

        Returns:
            norma L2
        """
        return math.sqrt(sum(v ** 2 for v in values))

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """
        Calcola la cosine similarity tra due gradienti.

        Interpretazione:
        -  1.0: gradienti identici (direzione uguale)
        -  0.0: gradienti ortogonali (nessuna correlazione)
        - -1.0: gradienti opposti (possibile poisoning)

        Args:
            a: primo gradiente (lista float)
            b: secondo gradiente (lista float)

        Returns:
            cosine similarity in [-1.0, 1.0]
        """
        if not a or not b:
            return 0.0

        # Allinea le lunghezze
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x ** 2 for x in a))
        norm_b = math.sqrt(sum(y ** 2 for y in b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (norm_a * norm_b)

    @classmethod
    def cluster_cosine_analysis(
        cls,
        node_gradients: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        """
        Calcola la cosine similarity media di ogni nodo vs tutti gli altri.

        Un nodo con similarità media bassa (< 0.5) è geometricamente
        isolato dal cluster — segnale di model poisoning o Byzantine fault.

        Args:
            node_gradients: {node_id: gradient_dict}

        Returns:
            {node_id: avg_cosine_similarity} in [-1.0, 1.0]
        """
        flat_grads = {
            node_id: cls.flatten(grad)
            for node_id, grad in node_gradients.items()
        }

        node_ids = list(flat_grads.keys())
        avg_similarities: dict[str, float] = {}

        for node_id in node_ids:
            similarities = [
                cls.cosine_similarity(flat_grads[node_id], flat_grads[other])
                for other in node_ids
                if other != node_id
            ]
            avg_similarities[node_id] = (
                sum(similarities) / len(similarities)
                if similarities else 1.0
            )

        return avg_similarities


# ─── Krum Detector ────────────────────────────────────────────────────────────

class KrumDetector:
    """
    Krum Byzantine fault detector.

    Krum assegna a ogni nodo uno score pari alla somma delle distanze
    ai suoi n-f-2 vicini più prossimi (dove f = Byzantine tolerance).

    Un nodo Byzantine ha gradienti molto distanti da tutti gli altri
    → alto Krum score → identificato come sospetto.

    Garanzia teorica: Krum è robusto a f nodi Byzantine su n totali,
    con n >= 2f+3.

    Riferimento: Blanchard et al., NeurIPS 2017
    """

    @staticmethod
    def compute_scores(
        node_gradients: dict[str, dict[str, Any]],
        byzantine_tolerance: int = 1,
    ) -> dict[str, float]:
        """
        Calcola il Krum score per ogni nodo.

        Score alto = nodo isolato = più sospetto.
        Score normalizzato in [0.0, 1.0] per confrontabilità.

        Args:
            node_gradients:      {node_id: gradient}
            byzantine_tolerance: numero massimo di nodi Byzantine (f)

        Returns:
            {node_id: normalized_krum_score}
        """
        flat_grads = {
            node_id: GradientAnalyzer.flatten(grad)
            for node_id, grad in node_gradients.items()
        }

        node_ids = list(flat_grads.keys())
        n = len(node_ids)
        f = byzantine_tolerance

        # Krum richiede almeno 2f+3 nodi
        if n < 2 * f + 3:
            return {node_id: 0.0 for node_id in node_ids}

        # Calcola distanze euclidee al quadrato tra tutti i nodi
        distances: dict[str, dict[str, float]] = {
            node_id: {} for node_id in node_ids
        }

        for i, node_i in enumerate(node_ids):
            for j, node_j in enumerate(node_ids):
                if i != j:
                    gi = flat_grads[node_i]
                    gj = flat_grads[node_j]
                    min_len = min(len(gi), len(gj))
                    dist = sum(
                        (a - b) ** 2
                        for a, b in zip(gi[:min_len], gj[:min_len])
                    )
                    distances[node_i][node_j] = dist

        # Krum score: somma delle n-f-2 distanze più piccole
        neighbors = n - f - 2
        krum_scores: dict[str, float] = {}

        for node_id in node_ids:
            sorted_dists = sorted(distances[node_id].values())
            krum_scores[node_id] = sum(sorted_dists[:neighbors])

        # Normalizza in [0.0, 1.0]
        max_score = max(krum_scores.values()) if krum_scores else 1.0
        if max_score > 0:
            krum_scores = {
                node_id: round(score / max_score, 4)
                for node_id, score in krum_scores.items()
            }

        return krum_scores

    @staticmethod
    def detect_byzantine(
        krum_scores: dict[str, float],
        threshold: float = 0.8,
    ) -> list[str]:
        """
        Identifica nodi Byzantine dai Krum scores.

        Args:
            krum_scores: {node_id: normalized_krum_score}
            threshold:   score oltre cui il nodo è sospetto (default 0.8)

        Returns:
            lista di node_id sospetti
        """
        return [
            node_id
            for node_id, score in krum_scores.items()
            if score > threshold
        ]


# ─── ChargingIDS ──────────────────────────────────────────────────────────────

class ChargingIDS(AbstractIDS):
    """
    IDS reale per reti di colonnine EV in ambiente FL.

    Integra quattro meccanismi di rilevamento:
    1. CUSUM       → deriva statistica nel tempo (singolo nodo)
    2. Krum        → Byzantine fault detection (livello cluster)
    3. Cosine sim  → model poisoning detection (livello cluster)
    4. FedMIA      → membership inference detection (livello cluster)

    I meccanismi 2, 3, 4 richiedono i gradienti di tutti i nodi
    → disponibili solo in analyze_round().

    Il meccanismo 1 opera su singolo AuditReport
    → disponibile in analyze().

    Uso tipico nel FLAREConnector:
        ids = ChargingIDS()
        # Per ogni nodo (analisi singola):
        alert = ids.analyze(audit_report)
        # Per il round completo (analisi cluster):
        round_analysis = ids.analyze_round(round_id, reports, gradients)
        excluded = round_analysis.byzantine_nodes
    """

    def __init__(
        self,
        config_path: str = "config/auditor.yaml",
        byzantine_tolerance: int = 1,
        cosine_threshold: float = 0.3,
        krum_threshold: float = 0.8,
        fedmia: FedMIA | None = None,
    ):
        """
        Inizializza ChargingIDS con tutti i detector.

        Args:
            config_path:          percorso config YAML
            byzantine_tolerance:  nodi Byzantine tollerati (f per Krum)
            cosine_threshold:     soglia cosine similarity (sotto = sospetto)
            krum_threshold:       soglia Krum score (sopra = sospetto)
            fedmia:               istanza FedMIA opzionale per MIA detection
        """
        config = _load_ids_config(config_path)

        self._alert_threshold: float = config["alert_threshold"]
        self._byzantine_tolerance = byzantine_tolerance
        self._cosine_threshold = cosine_threshold
        self._krum_threshold = krum_threshold

        # CUSUM per privacy score e epsilon di ogni nodo
        self._cusum_privacy = CUSUMDetector(threshold=5.0, drift=0.3)
        self._cusum_epsilon = CUSUMDetector(threshold=5.0, drift=0.1)

        # Analyzer per operazioni geometriche sui gradienti
        self._gradient_analyzer = GradientAnalyzer()

        # Krum detector per Byzantine faults
        self._krum = KrumDetector()

        # FedMIA opzionale — se None, MIA detection disabilitata
        self._fedmia = fedmia

        # Baseline EMA per ogni nodo
        # {node_id: {"privacy_score": float, "epsilon": float, "count": int}}
        self._baselines: dict[str, dict[str, Any]] = {}

        # Risk score per ogni nodo [0.0, 1.0]
        self._risk_scores: dict[str, float] = {}

        # Storico alert per analisi post-esperimento
        self._alert_history: list[IDSAlert] = []

        # Storico analisi di round per il paper
        self._round_history: list[RoundAnalysis] = []

    def analyze(self, report: AuditReport) -> IDSAlert | None:
        """
        Analisi singolo nodo basata su AuditReport.

        Detector attivi: regole esplicite + CUSUM su privacy score ed epsilon.

        Args:
            report: AuditReport del PrivacyAuditor

        Returns:
            IDSAlert se anomalia rilevata, None altrimenti
        """
        reasons: list[str] = []
        severity = "LOW"
        action = "MONITOR"

        # ── Regole esplicite sulle minacce del PrivacyAuditor ──────────
        if "GRADIENT_EXPLOSION" in report.threats_detected:
            reasons.append(
                "Gradient explosion — possibile model poisoning attack"
            )
            severity = "CRITICAL"
            action = "EXCLUDE"

        if "PRIVACY_BUDGET_EXHAUSTED" in report.threats_detected:
            reasons.append("Budget privacy esaurito — alto rischio MIA")
            if severity != "CRITICAL":
                severity = "HIGH"
                action = "EXCLUDE"

        if "PRIVACY_BUDGET_NEAR_EXHAUSTION" in report.threats_detected:
            reasons.append("Budget privacy quasi esaurito")
            if severity not in ("CRITICAL", "HIGH"):
                severity = "MEDIUM"
                action = "THROTTLE"

        if "FEDMIA_SUSPICIOUS_LOW_SENSITIVITY" in report.threats_detected:
            reasons.append(
                "Pattern FedMIA — sensitivity anomalmente bassa"
            )
            if severity not in ("CRITICAL", "HIGH"):
                severity = "MEDIUM"
                action = "THROTTLE"

        # ── CUSUM su privacy score ──────────────────────────────────────
        cusum_privacy_alarm = self._cusum_privacy.update(
            report.node_id, report.privacy_score
        )
        if cusum_privacy_alarm:
            reasons.append(
                f"CUSUM: deriva statistica nel privacy score rilevata "
                f"(score corrente: {report.privacy_score:.3f})"
            )
            if severity == "LOW":
                severity = "MEDIUM"
                action = "THROTTLE"

        # ── CUSUM su epsilon ────────────────────────────────────────────
        cusum_epsilon_alarm = self._cusum_epsilon.update(
            report.node_id, report.epsilon
        )
        if cusum_epsilon_alarm:
            reasons.append(
                f"CUSUM: deriva statistica nell'epsilon consumato "
                f"(epsilon corrente: {report.epsilon:.6f})"
            )
            if severity == "LOW":
                severity = "MEDIUM"
                action = "THROTTLE"

        # ── Aggiorna risk score e baseline ──────────────────────────────
        self._update_risk_score(report.node_id, len(reasons))
        self.update_baseline(report.node_id, report)

        if not reasons:
            return None

        alert = IDSAlert(
            node_id=report.node_id,
            round_id=report.round_id,
            severity=severity,
            reasons=reasons,
            recommended_action=action,
            metadata={
                "privacy_score": report.privacy_score,
                "epsilon": report.epsilon,
                "risk_score": self._risk_scores.get(report.node_id, 0.0),
                "cusum_privacy": self._cusum_privacy.get_cusum_values(
                    report.node_id
                ),
                "cusum_epsilon": self._cusum_epsilon.get_cusum_values(
                    report.node_id
                ),
            },
        )
        self._alert_history.append(alert)
        return alert

    def analyze_round(
        self,
        round_id: int,
        reports: dict[str, AuditReport],
        gradients: dict[str, dict[str, Any]],
    ) -> RoundAnalysis:
        """
        Analisi completa del round FL con tutti i detector.

        Fasi:
        1. analyze() per ogni nodo (CUSUM + regole)
        2. Krum Byzantine detection sul cluster completo
        3. Cosine similarity analysis sul cluster completo
        4. FedMIA cluster attack (se shadow model addestrato)
        5. Aggrega risultati in RoundAnalysis

        Args:
            round_id:  numero del round FL
            reports:   {node_id: AuditReport} per tutti i nodi
            gradients: {node_id: model_update} per tutti i nodi

        Returns:
            RoundAnalysis con tutti i risultati
        """
        all_alerts: list[IDSAlert] = []

        # Step 1: analisi singolo nodo per tutti i nodi
        for node_id, report in reports.items():
            alert = self.analyze(report)
            if alert is not None:
                all_alerts.append(alert)

        # Step 2: Krum Byzantine detection
        krum_scores = self._krum.compute_scores(
            gradients,
            byzantine_tolerance=self._byzantine_tolerance,
        )
        byzantine_nodes = self._krum.detect_byzantine(
            krum_scores,
            threshold=self._krum_threshold,
        )

        # Genera alert per nodi Byzantine
        for node_id in byzantine_nodes:
            score = krum_scores.get(node_id, 0.0)
            alert = IDSAlert(
                node_id=node_id,
                round_id=round_id,
                severity="HIGH",
                reasons=[
                    f"Krum Byzantine detection: score={score:.4f} "
                    f"(soglia={self._krum_threshold}) — "
                    f"gradiente geometricamente isolato dal cluster"
                ],
                recommended_action="EXCLUDE",
                metadata={
                    "krum_score": score,
                    "krum_threshold": self._krum_threshold,
                    "detector": "Krum",
                },
            )
            all_alerts.append(alert)
            self._alert_history.append(alert)

        # Step 3: Cosine similarity analysis
        cosine_scores = self._gradient_analyzer.cluster_cosine_analysis(
            gradients
        )
        low_similarity_nodes = [
            node_id
            for node_id, sim in cosine_scores.items()
            if sim < self._cosine_threshold
        ]

        # Genera alert per nodi con bassa cosine similarity
        for node_id in low_similarity_nodes:
            sim = cosine_scores.get(node_id, 1.0)
            alert = IDSAlert(
                node_id=node_id,
                round_id=round_id,
                severity="HIGH",
                reasons=[
                    f"Cosine similarity anomala: {sim:.4f} "
                    f"(soglia={self._cosine_threshold}) — "
                    f"gradiente ortogonale/opposto al cluster"
                ],
                recommended_action="EXCLUDE",
                metadata={
                    "cosine_similarity": sim,
                    "cosine_threshold": self._cosine_threshold,
                    "detector": "CosineSimilarity",
                },
            )
            all_alerts.append(alert)
            self._alert_history.append(alert)

        # Step 4: FedMIA cluster attack (opzionale)
        if self._fedmia is not None:
            try:
                mia_results = self._fedmia.run_cluster_attack(
                    cluster_id="all",
                    round_id=round_id,
                    cluster_gradients=gradients,
                )
                for result in mia_results:
                    if result.is_member and result.confidence > 0.7:
                        alert = IDSAlert(
                            node_id=result.node_id,
                            round_id=round_id,
                            severity="MEDIUM",
                            reasons=[
                                f"FedMIA: membership score={result.membership_score:.4f} "
                                f"confidence={result.confidence:.4f} — "
                                f"dati di training potenzialmente inferibili"
                            ],
                            recommended_action="THROTTLE",
                            metadata={
                                "membership_score": result.membership_score,
                                "confidence": result.confidence,
                                "detector": "FedMIA",
                            },
                        )
                        all_alerts.append(alert)
                        self._alert_history.append(alert)
            except RuntimeError:
                # Shadow model non ancora addestrato — skip MIA detection
                pass

        # Step 5: aggiorna risk scores per tutti i nodi del round
        all_detected = set(byzantine_nodes) | set(low_similarity_nodes)
        for node_id in gradients:
            anomalies = 1 if node_id in all_detected else 0
            self._update_risk_score(node_id, anomalies)

        round_analysis = RoundAnalysis(
            round_id=round_id,
            alerts=all_alerts,
            byzantine_nodes=byzantine_nodes,
            low_similarity_nodes=low_similarity_nodes,
            krum_scores=krum_scores,
            cosine_scores=cosine_scores,
        )
        self._round_history.append(round_analysis)
        return round_analysis

    def update_baseline(self, node_id: str, report: AuditReport) -> None:
        """
        Aggiorna la baseline con Exponential Moving Average (EMA, alpha=0.3).

        EMA con alpha=0.3:
        - Risponde ai cambiamenti ma non è dominata dagli outlier
        - Standard per monitoraggio FL con round variabili

        Args:
            node_id: identificatore del nodo
            report:  AuditReport del round corrente
        """
        alpha = 0.3

        if node_id not in self._baselines:
            self._baselines[node_id] = {
                "privacy_score": report.privacy_score,
                "epsilon": report.epsilon,
                "count": 1,
            }
        else:
            b = self._baselines[node_id]
            b["privacy_score"] = (
                alpha * report.privacy_score +
                (1 - alpha) * b["privacy_score"]
            )
            b["epsilon"] = (
                alpha * report.epsilon +
                (1 - alpha) * b["epsilon"]
            )
            b["count"] += 1

    def _update_risk_score(self, node_id: str, anomaly_count: int) -> None:
        """
        Aggiorna il risk score del nodo.

        Aumenta di 0.2 per ogni anomalia rilevata.
        Decade del 10% per round senza anomalie (riabilitazione).
        Clamped in [0.0, 1.0].

        Args:
            node_id:       identificatore del nodo
            anomaly_count: numero anomalie rilevate nel round
        """
        current = self._risk_scores.get(node_id, 0.0)
        if anomaly_count > 0:
            new_score = min(1.0, current + anomaly_count * 0.2)
        else:
            new_score = current * 0.9
        self._risk_scores[node_id] = round(new_score, 4)

    def get_node_risk_score(self, node_id: str) -> float:
        """
        Restituisce il risk score corrente del nodo [0.0, 1.0].

        Args:
            node_id: identificatore del nodo

        Returns:
            risk score (0.0 se nodo mai visto)
        """
        return self._risk_scores.get(node_id, 0.0)

    def get_alert_history(self) -> list[IDSAlert]:
        """
        Restituisce lo storico completo degli alert.
        Utile per analisi post-esperimento e per il paper.

        Returns:
            lista di IDSAlert in ordine cronologico
        """
        return self._alert_history.copy()

    def get_round_history(self) -> list[RoundAnalysis]:
        """
        Restituisce lo storico delle analisi di round.

        Returns:
            lista di RoundAnalysis in ordine cronologico
        """
        return self._round_history.copy()

    def reset(self) -> None:
        """
        Resetta lo stato interno per un nuovo esperimento.
        """
        self._cusum_privacy.reset()
        self._cusum_epsilon.reset()
        self._baselines.clear()
        self._risk_scores.clear()
        self._alert_history.clear()
        self._round_history.clear()
