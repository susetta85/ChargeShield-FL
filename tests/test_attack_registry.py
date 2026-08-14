# tests/test_attack_registry.py
"""
Test per src/core/base_attack.py + src/plugins/attacks/ — l'interfaccia
"pluggable attack" implementata realmente il 2026-07-24, su richiesta
esplicita dell'utente (vedi docs/DSN2027_Positioning.md e
docs/DeveloperGuide.md, entrambi corretti lo stesso giorno perché
descrivevano questo meccanismo come già esistente quando non lo era).

Questi test coprono SOLO il livello di interfaccia/registro (BaseAttack,
ATTACK_REGISTRY, le tre classi wrapper) — non torch-dipendente, quindi
eseguibile davvero in questo sandbox, a differenza delle funzioni che
ciascuna classe richiama (run_fedmia/run_fedmia_shadow/run_lira in
scripts/run_experiments.py, che importa torch a livello di modulo — vedi
tests/test_run_experiments_integration.py per quelle, non eseguibili qui).
Non testano quindi che YeomAttack/ShadowAttack/LiRAAttack.run() produca i
numeri corretti (già coperto altrove, invariato) — solo che la struttura del
registro sia corretta: nomi giusti, sottoclassamento corretto, firma di run()
conforme al contratto.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Necessario SOLO per l'ultimo test (signature check): la classe wrapper
# importa run_experiments in modo lazy dentro run() (vedi src/plugins/attacks/
# yeom.py per il perché), e run_experiments.py vive in scripts/, non in src/.
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from core.base_attack import BaseAttack  # noqa: E402
from plugins.attacks import (  # noqa: E402
    ATTACK_REGISTRY,
    LiRAAttack,
    ShadowAttack,
    YeomAttack,
)


class TestAttackRegistry:
    def test_registry_has_exactly_the_three_validated_attacks(self):
        assert set(ATTACK_REGISTRY.keys()) == {"yeom", "shadow", "lira"}, (
            "il registro deve contenere esattamente yeom/shadow/lira — se questo "
            "test fallisce dopo aver aggiunto un nuovo attacco, aggiornalo di "
            "proposito (non un bug)"
        )

    def test_registry_maps_to_correct_classes(self):
        assert ATTACK_REGISTRY["yeom"] is YeomAttack
        assert ATTACK_REGISTRY["shadow"] is ShadowAttack
        assert ATTACK_REGISTRY["lira"] is LiRAAttack

    @pytest.mark.parametrize("attack_name", ["yeom", "shadow", "lira"])
    def test_each_registered_class_is_a_real_baseattack_subclass(self, attack_name):
        """Regressione diretta per l'overclaim corretto in
        docs/DSN2027_Positioning.md/docs/DeveloperGuide.md il 2026-07-24: prima
        di questo fix NON esisteva nessuna classe BaseAttack nel codice — questo
        test fallirebbe con un semplice file di funzioni sciolte."""
        cls = ATTACK_REGISTRY[attack_name]
        assert issubclass(cls, BaseAttack)
        instance = cls()
        assert instance.name == attack_name

    def test_baseattack_cannot_be_instantiated_directly(self):
        """È un contratto astratto (ABC), non una classe concreta utilizzabile
        da sola — run() non ha implementazione di default."""
        with pytest.raises(TypeError):
            BaseAttack()  # type: ignore[abstract]

    def test_incomplete_subclass_without_run_cannot_be_instantiated(self):
        """Una sottoclasse che non implementa run() deve restare astratta —
        garantisce che il contratto sia enforced, non solo documentato."""

        class IncompleteAttack(BaseAttack):
            name = "incomplete"

        with pytest.raises(TypeError):
            IncompleteAttack()  # type: ignore[abstract]

    def test_run_method_signature_accepts_common_contract_and_kwargs(self):
        """Verifica che ogni classe registrata accetti la firma comune
        (cfg, train_sessions, holdout_sessions, fl_results, **kwargs) senza
        sollevare TypeError in fase di binding degli argomenti.

        FIX 2026-08-12 (trovato eseguendo per la prima volta la suite intera
        su una macchina CON torch installato, non solo in questo sandbox):
        la versione precedente invocava davvero instance.run(...) e si
        aspettava un ModuleNotFoundError("torch") dall'import lazy dentro
        run_fedmia()/run_lira() — assunzione valida SOLO in un ambiente senza
        torch (questo sandbox). Su una macchina con torch installato (l'uso
        normale/reale del progetto) l'import lazy riesce, l'esecuzione
        prosegue nel corpo della funzione e fallisce invece con un
        KeyError('ml') dovuto al cfg={} volutamente vuoto passato dal test —
        un errore vero ma NON quello che questo test intende verificare (la
        sola forma della firma, non il comportamento a runtime). Fix: usare
        inspect.signature(...).bind() per validare che gli argomenti si
        leghino correttamente ai parametri SENZA eseguire il corpo di run()
        — corretto e stabile indipendentemente dalla presenza di torch,
        sandbox o macchina reale che sia.
        """
        import inspect

        for attack_name in ("yeom", "shadow", "lira"):
            instance = ATTACK_REGISTRY[attack_name]()
            sig = inspect.signature(instance.run)
            # Solleva TypeError se la firma non accetta questi argomenti
            # (posizionali/keyword richiesti dal contratto comune, più
            # eventuali kwargs extra che devono finire in **kwargs).
            sig.bind(
                cfg={}, train_sessions=[], holdout_sessions=[], fl_results={},
                n_shadow=8, shadow_epochs_cap=None, no_dp=False,
                dp_mode="dp-fedavg", cluster_membership=None,
            )
