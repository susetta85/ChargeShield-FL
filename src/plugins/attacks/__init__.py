# src/plugins/attacks/__init__.py
"""
Registro degli attacchi MIA "pluggable" — implementato 2026-07-24 (vedi
src/core/base_attack.py per il razionale completo e i vincoli di design).

Prima di questo fix questo file era vuoto (0 byte) — nessun meccanismo di
registrazione esisteva, nonostante docs/DeveloperGuide.md e
docs/DSN2027_Positioning.md lo dessero per scontato (correzione 2026-07-24
in entrambi i documenti).

Per aggiungere un nuovo attacco:
  1. Crea src/plugins/attacks/<nome>.py con una classe che implementa
     BaseAttack (vedi yeom.py/shadow.py/lira.py come esempio — per un
     attacco realmente nuovo, non un wrapper su codice esistente, implementa
     run() direttamente invece di delegare a una funzione in
     scripts/run_experiments.py).
  2. Aggiungila ad ATTACK_REGISTRY sotto.
Non serve toccare scripts/run_experiments.py::main() — itera ATTACK_REGISTRY
(vedi il blocco "Esegue tutti gli attacchi registrati" in main()).
"""

from __future__ import annotations

from core.base_attack import BaseAttack
from plugins.attacks.lira import LiRAAttack
from plugins.attacks.shadow import ShadowAttack
from plugins.attacks.yeom import YeomAttack

#: nome → classe (non istanza: ogni chiamata a main() ne crea una nuova,
#: coerente con run_fedmia()/run_fedmia_shadow()/run_lira() che erano già
#: funzioni stateless — nessuna delle tre classi qui tiene stato fra round).
ATTACK_REGISTRY: dict[str, type[BaseAttack]] = {
    YeomAttack.name: YeomAttack,
    ShadowAttack.name: ShadowAttack,
    LiRAAttack.name: LiRAAttack,
}

__all__ = ["BaseAttack", "ATTACK_REGISTRY", "YeomAttack", "ShadowAttack", "LiRAAttack"]
