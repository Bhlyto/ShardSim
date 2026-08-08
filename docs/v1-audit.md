# Audit ShardSim vers une V1 taggable

Date de l’audit : 8 août 2026. Source de vérité : état de travail local du dépôt,
pas la documentation historique.

## 1. Verdict

ShardSim est réellement aujourd’hui un ensemble Python centré sur la diffusion
thermique transitoire 2D. Son composant le plus fiable est un solveur explicite par
différences finies, déterministe, avec conditions de Dirichlet. Autour de ce noyau se
trouvent une adaptation OpenFOAM, un pipeline multi-fidélité, des datasets vérifiés,
du raffinement local, plusieurs surrogates ML et un gestionnaire de campagnes.

Ce n’est pas un framework physique générique stable : un seul domaine est implémenté,
et les abstractions génériques servent presque exclusivement `heat-2d`. Ce n’est pas
non plus un scheduler distribué dans l’état de travail Python. Les concepts MPI,
orchestrateur et décision distribuée appartiennent à l’ancienne architecture C++
visible dans l’historique distant, pas au noyau exécutable local audité.

Le code scientifique interne est sain : 41 tests sont exécutés avec succès. Quatre
tests optionnels sont ignorés (trois CNN/PyTorch, un OpenFOAM/Docker). Après la
stabilisation V1, le flux minimal scénario → calcul → artifacts est couvert, y compris
100 exécutions indépendantes.

Au début de l’audit, le dépôt n’était pas taggable : la branche locale était en retard
de huit commits sur `origin/main` et presque toute l’implémentation Python était non
suivie. La base Python a depuis été isolée sur `codex/v1-core`; l’ancienne architecture
C++/MPI n’est volontairement pas fusionnée dans la V1.

## 2. Architecture actuelle reconstruite

```text
Entrées générées par code                 campaign.json + cases.jsonl
          │                                         │
          ▼                                         ▼
 contracts + domains/heat2d             campaign/core.py (1 822 lignes)
          │                                │       │       │
          ├──────────────┐                 │       │       └─ reporting HTML/CSV/NPZ
          ▼              ▼                 ▼       ▼
 solver thermique    adapter OpenFOAM   datasets  surrogates ML
          │              │                 │       │
          └──────┬───────┘                 └───┬───┘
                 ▼                             ▼
          SimulationResult             preview/workflow/active learning
```

Le flux V1 stabilisé est plus court :

```text
scenario.json
      ▼
scenario.py (validation stricte + adaptation vers SimulationCase)
      ▼
execution.py → HeatEquationSolver
      ▼
scenario.json + field.npy + run.log + result.json
```

Les exécutables sont désormais `shardsim`/`python -m shardsim` pour la V1 et
`shardsim-campaign` pour l’expérimental historique.

## 3. Bloquants P0

| Problème | Zone | Impact | Proposition | Bénéfice | Coût | Risque | État |
|---|---|---|---|---|---|---|---|
| Aucun contrat scénario/résultat simple | `scenario.py`, `execution.py`, `cli.py` absents avant audit | Impossible à appeler proprement depuis BHCScheduler | Schémas V1 stricts, artifacts et checksums | Processus autonome et reproductible | Moyen | Faible | Corrigé |
| API racine exposait toute la pile avancée | `src/shardsim/__init__.py` | Surface de compatibilité ingérable | Limiter l’API stable au scénario, exécution, contrats et solveur interne | V1 maintenable seul | Faible | Rupture avant V1 | Corrigé |
| README décrivait ChatMPI, du distribué et des scripts absents | ancien `README.md` | On ne pouvait pas savoir comment lancer le code réel | Remplacer par le parcours V1 vérifié | Onboarding fiable | Faible | Faible | Corrigé |
| Aucun test E2E de processus ni de lot 100 | `tests/test_v1_acceptance.py` | Contrat scheduler non démontré | Tester CLI externe, reproductibilité et 100 sorties isolées | Critères d’acceptation automatisés | Faible | Faible | Corrigé |
| Deux architectures divergentes et V1 non suivie par Git | branche locale initiale, `origin/main` | Un tag aurait perdu le code V1 ou réintroduit MPI/orchestration | Isoler la base Python sur `codex/v1-core` sans fusion automatique | Historique et tag fiables | Moyen | Élevé si fusion automatique | Corrigé |
| Ancien environnement virtuel local cassé | `.venv` local ignoré | La commande historique de test échoue sur cette machine | Construire le wheel et tester une installation vierge | Reproductibilité développeur | Faible | Faible | Corrigé par un smoke test isolé |

## 4. KEEP / SIMPLIFY / REMOVE / LATER

| Décision | Composants | Motif |
|---|---|---|
| KEEP | `scenario.py`, `execution.py`, `cli.py`, `contracts.py`, `domains/heat2d.py`, `solvers/heat.py`, `interpolation.py`, métriques thermiques | Chemin direct scénario → physique → résultat |
| KEEP | benchmark de solution manufacturée dans `verification/heat.py` | Test représentatif du domaine réel |
| SIMPLIFY | `canonical.py` | V1 utilise surtout `FieldLocation`; maillage et champs génériques doivent rester internes tant qu’un second backend stable ne les justifie pas |
| SIMPLIFY | helpers JSON/hash/écriture atomique dupliqués dans `dataset.py`, `campaign/core.py`, `campaign/reporting.py`, `execution.py` | Mutualiser seulement lors d’un prochain changement de ces modules, sans refactor transversal risqué avant le tag |
| SIMPLIFY | `campaign/core.py` (1 822 lignes) | Mélange design d’expérience, exécution, entraînement, registre de modèles, évaluation et environnement |
| LATER | `pipeline.py`, `preview.py`, `dataset.py`, `design.py`, `workflow.py`, `active_learning.py` | Cohérents entre eux et testés, mais non nécessaires à une exécution physique V1 |
| LATER | `adaptive.py`, `refinement.py` | Optimisation utile après stabilisation du calcul nominal |
| LATER | `solvers/openfoam.py` et `canonical.py` complet | Dépendance Docker et test d’intégration opt-in ; candidat V1.1 après CI dédiée |
| LATER | `surrogates/*`, particulièrement `heat_unet.py` | La V1 doit fonctionner sans IA ; trois tests CNN ne s’exécutent pas dans l’environnement minimal |
| REMOVE du cœur | ancien C++ MPI, orchestrateur, décision distribuée et scheduling présents dans `origin/main` | Responsabilité de BHCScheduler et architecture concurrente |
| REMOVE de la livraison | `instruction.md`, logs/background et campagnes générées | Documents d’intention ou état d’exécution, pas du produit V1 |

## 5. Architecture cible minimale

```text
Scenario + Validation
        ▼
Model heat-2d + Solver interne
        ▼
Execution + Result serialization
        ▼
CLI minimale
```

Il n’y a pas de registre de plugins, d’orchestrateur, de queue ni de couche ML dans
le chemin V1.

## 6. Plan de migration incrémental

1. **Fait** — figer le schéma scénario/résultat 1.0 et la CLI minimale.
2. **Fait** — écrire les artifacts atomiquement, enregistrer versions, environnement,
   paramètres, seed, durée, code de sortie et checksums.
3. **Fait** — isoler l’API racine et documenter tout le reste comme expérimental.
4. **Fait** — automatiser les dix critères d’acceptation et conserver le benchmark
   thermique manufacturé existant.
5. **Fait** — isoler la base Python sur `codex/v1-core` sans fusion automatique de
   l’ancienne architecture distribuée et sélectionner explicitement les fichiers V1.
6. **Fait localement** — construire le wheel, l’installer dans un environnement vierge
   et exécuter `validate`, `run` et `inspect` depuis hors du dépôt.
7. **Ajouté, à confirmer sur GitHub** — CI minimale Python 3.10 et 3.13 ; tagger
   `v1.0.0` uniquement après ses premiers passages verts.
8. **Après tag** — déplacer physiquement les modules avancés sous un espace
   expérimental lors d’une version mineure, avec dépréciation des anciens imports.

Tags intermédiaires suggérés : `v1.0.0-rc1` après résolution Git et CI verte, puis
`v1.0.0` après installation et smoke test sur un checkout propre.

## 7. Définition exacte de ShardSim V1

**ShardSim V1 exécute localement un scénario versionné de diffusion thermique 2D de
façon déterministe et produit un champ, un journal et un résultat structuré dont la
provenance et l’intégrité sont vérifiables.**

## 8. Tests d’acceptation V1

| Critère | Test automatisé |
|---|---|
| Scénario valide chargé | `test_valid_scenario_loads_and_invalid_scenario_is_explicit` |
| Scénario invalide rejeté | même test + `test_invalid_cli_input_returns_nonzero_with_clear_error` |
| Simulation simple exécutée | `test_run_saves_structured_reproducible_result_and_logs` |
| Résultat sauvegardé | même test, présence et inspection de `result.json`/`field.npy` |
| Erreurs explicites | test CLI invalide, message sur le chemin JSON fautif |
| Logs exploitables | test de `run.log` avec scénario, statut, steps et durée |
| Reproductibilité | deux champs identiques et même `reproducibility_key` |
| 100 scénarios indépendants | `test_one_hundred_scenarios_run_independently` |
| Lancement depuis script externe | `test_cli_is_usable_as_a_plain_external_process` |
| Compatible BHCScheduler | même test : un processus, un code de sortie, un répertoire |
| Physique représentative | `test_internal_verification_converges` sur solution manufacturée |

## 9. Fonctionnalités repoussées après V1

- OpenFOAM comme backend officiellement supporté ;
- entraînement, promotion et registre de modèles ;
- U-Net et toute dépendance PyTorch ;
- génération de campagnes Latin Hypercube ;
- active learning, preview et raffinement adaptatif ;
- dashboards et exports de campagne ;
- autres domaines physiques et autres formats d’entrée ;
- MPI, distribution, workers, heartbeat, retries, leases, allocation et scheduling ;
- UI, cloud, Kubernetes et système de plugins.

Ces reports sont des décisions de périmètre, pas des promesses pour la V2.
