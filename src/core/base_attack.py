# src/core/base_attack.py
"""
Contratto comune per gli attacchi di Membership Inference (e, in futuro,
Gradient Inversion / Property Inference) contro fl_results — vedi
docs/DeveloperGuide.md §5.4 e docs/DSN2027_Positioning.md.

Implementato 2026-07-24 su richiesta esplicita dell'utente, per rendere reale
(non solo "progettata") l'interfaccia pluggable descritta in quei documenti.
Prima di questo fix non esisteva nessuna classe base per gli attacchi in
`src/` — `src/plugins/attacks/` conteneva solo `fedmia.py`, un file isolato,
mai chiamato dalla pipeline reale, senza classe base né registro (vedi la
correzione 2026-07-24 in docs/DSN2027_Positioning.md e docs/DeveloperGuide.md
per la storia completa di questo gap).

Vincolo di design — "non toccare la logica già validata": Yeom (`run_fedmia`),
Shadow (`run_fedmia_shadow`) e LiRA (`run_lira`), tutti in
`scripts/run_experiments.py`, hanno ciascuno diversi round di fix empirici
documentati nelle rispettive docstring (LiRA in particolare: 5 round di fix
trovati SOLO eseguendo davvero il codice — vedi la sua docstring). Riscriverli
come metodi di classe da zero avrebbe rischiato di reintrodurre uno di quei
bug, senza modo di verificarlo in questo sandbox (torch non installabile qui).
Le classi in `src/plugins/attacks/` che implementano `BaseAttack` sono quindi
wrapper sottili: `run()` per ciascun attacco chiama la funzione esistente,
invariata, con gli stessi argomenti — zero cambio di comportamento numerico,
solo un'interfaccia comune sopra funzioni già validate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAttack(ABC):
    """Contratto comune per un attacco MIA eseguibile contro fl_results.

    Un nuovo attacco si aggiunge implementando questa classe in un nuovo file
    sotto `src/plugins/attacks/` e registrandola in
    `src/plugins/attacks/ATTACK_REGISTRY` (vedi `src/plugins/attacks/__init__.py`)
    — non serve toccare `scripts/run_experiments.py`'s main(), che itera il
    registro invece di chiamare ogni attacco per nome (vedi il blocco
    "Esegue tutti gli attacchi registrati" in main()).
    """

    #: Nome breve, usato come chiave nel registro e nei log/report.
    name: str = "unnamed"

    @abstractmethod
    def run(
        self,
        cfg: dict,
        train_sessions: list[dict[str, Any]],
        holdout_sessions: list[dict[str, Any]],
        fl_results: dict[int, dict[str, Any]],
        **kwargs: Any,
    ) -> dict[int, dict[str, Any]]:
        """Esegue l'attacco e ritorna {round_num: {..metriche per round..}}.

        Contratto: stesso schema di ritorno di run_fedmia()/run_fedmia_shadow()/
        run_lira() in scripts/run_experiments.py — un dict indicizzato per
        round, ciascuno con almeno un campo AUC-ROC (nome del campo specifico
        per attacco, es. "auc_roc"/"shadow_auc_roc"/"lira_auc_roc" — non
        unificato in v1 per non toccare lo schema JSON già consumato da
        generate_excel_report.py e dai file experiment_*.json esistenti).

        Args:
            cfg:              configurazione esperimento.
            train_sessions:   sessioni usate nel FL training (membri).
            holdout_sessions: sessioni mai viste durante FL (non-membri).
            fl_results:       dict round → {"global_weights"/"raw_updates"/..., ...}
                              prodotto da run_fl_rounds().
            **kwargs:         parametri opzionali specifici dell'attacco (es.
                              LiRA usa n_shadow/shadow_epochs_cap/no_dp/dp_mode/
                              cluster_membership). Ogni implementazione
                              legge solo le chiavi che le servono e ignora
                              il resto — permette a main() di passare un
                              unico set di kwargs a tutti gli attacchi
                              registrati senza if/else per attacco.
        """
        ...
