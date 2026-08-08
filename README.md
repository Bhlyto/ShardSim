# ShardSim

ShardSim V1 prend un scénario thermique 2D versionné, l’exécute localement de façon
reproductible, puis écrit un résultat structuré et vérifiable.

Le cœur V1 ne planifie pas de tâches, ne gère ni workers ni cluster, et ne dépend pas
du ML. Un ordonnanceur externe, notamment BHCScheduler, peut simplement lancer un
processus ShardSim par scénario et collecter son répertoire de sortie.

## Installation

ShardSim nécessite Python 3.10 ou plus récent et NumPy.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

## Interface V1

```powershell
shardsim validate examples/scenario.heat2d.json
shardsim run examples/scenario.heat2d.json --output results/heat-gaussian-demo
shardsim inspect results/heat-gaussian-demo/result.json
```

Sans installation du script console, l’interface équivalente est :

```powershell
$env:PYTHONPATH = "src"
python -m shardsim run examples/scenario.heat2d.json --output results/heat-gaussian-demo
```

Une exécution réussie crée exclusivement dans le répertoire demandé :

```text
results/heat-gaussian-demo/
├── scenario.json   scénario validé et normalisé
├── field.npy       champ de température final
├── run.log         journal d’exécution
└── result.json     statut, versions, métriques, environnement et checksums
```

ShardSim refuse d’écraser un résultat existant. Chaque scénario doit recevoir son
propre répertoire de sortie, ce qui rend l’appel sûr depuis un scheduler externe.

Le contrat complet est décrit dans [docs/scenario-v1.md](docs/scenario-v1.md). Le
verdict architectural, les décisions de périmètre et la feuille de route se trouvent
dans [docs/v1-audit.md](docs/v1-audit.md).

## Tests

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

La suite couvre le solveur thermique, les erreurs de validation, la reproductibilité,
la CLI en sous-processus et 100 scénarios successifs. Les tests CNN nécessitent
l’extra `cnn`; le test OpenFOAM est une intégration Docker explicitement opt-in. Ces
deux capacités ne font pas partie du contrat V1.

## Périmètre

Le seul modèle stable en V1 est `heat-2d`, résolu par le backend interne explicite.
Les modules `campaign`, `workflow`, `active_learning`, `adaptive`, `refinement`,
`surrogates` et le backend `openfoam` restent disponibles pour expérimentation via
leurs chemins Python explicites, mais ne bénéficient pas de la compatibilité V1.

Le script historique `shardsim-campaign` est conservé pour ne pas casser les essais
existants. Il n’est pas requis pour lancer un scénario V1.
