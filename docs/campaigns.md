# Campagnes reproductibles lancées manuellement

La couche de campagne conserve trois fichiers légers à versionner :

- `campaign.json` : espace d’exploration éditable, solveurs, maillages et seuils ;
- `campaign.lock.json` : empreintes du contrat et de la liste des cas ;
- `cases.jsonl` : valeurs numériques exactes de chaque cas et split associé.

Les datasets, journaux, modèles et rapports sont écrits sous
`outputs/<empreinte-du-contrat>/` et ne doivent pas être versionnés.

Chaque famille utilise un plan latin hypercube dérivé indépendamment du seed principal. Ajouter
une famille ne modifie donc pas les cas déjà verrouillés des autres familles. Le modèle par défaut
est le surrogate thermique résiduel local ; son environnement numérique exact est enregistré pour
permettre de reproduire son entraînement CPU.

## Première campagne

Depuis la racine du dépôt :

```powershell
$env:PYTHONPATH = "src"
python -m shardsim.campaign init campaigns/heat-v1 --name heat-v1 --seed 20260714
```

Éditer ensuite `campaigns/heat-v1/campaign.json`, puis verrouiller les valeurs exactes :

```powershell
python -m shardsim.campaign lock campaigns/heat-v1
git add campaigns/heat-v1/campaign.json campaigns/heat-v1/campaign.lock.json campaigns/heat-v1/cases.jsonl
```

Une modification, même silencieuse, de `campaign.json` ou `cases.jsonl` après cette étape bloque
l’exécution. Il faut relancer explicitement `lock --force`, ce qui crée une nouvelle clé de sortie.

## Exécutions manuelles et reprise

Toujours inspecter avant de lancer OpenFOAM :

```powershell
python -m shardsim.campaign list campaigns/heat-v1 --split train
python -m shardsim.campaign run campaigns/heat-v1 --split train --limit 3 --dry-run
```

Lancer ensuite un petit lot :

```powershell
python -m shardsim.campaign run campaigns/heat-v1 --split train --limit 3
python -m shardsim.campaign status campaigns/heat-v1
```

La même commande reprend aux prochains cas absents du manifeste. Chaque résultat est écrit
atomiquement avec un SHA-256. Chaque lancement produit aussi un journal comprenant la version de
Python, NumPy, ShardSim, le commit Git, l’état modifié du dépôt et l’image OpenFOAM épinglée.

Sélections plus fines :

```powershell
python -m shardsim.campaign run campaigns/heat-v1 --family train-centered --limit 2
python -m shardsim.campaign run campaigns/heat-v1 --case-id train-centered-0000-XXXXXXXXXX
python -m shardsim.campaign run campaigns/heat-v1 --all
```

`--all` est obligatoire pour lancer toute la campagne sans filtre, afin d’éviter un lancement
coûteux accidentel.

## Entraînement et évaluation

Par défaut, l’entraînement refuse un split incomplet :

```powershell
python -m shardsim.campaign train campaigns/heat-v1
python -m shardsim.campaign run campaigns/heat-v1 --split validation
python -m shardsim.campaign evaluate campaigns/heat-v1 --split validation
python -m shardsim.campaign run campaigns/heat-v1 --split test
python -m shardsim.campaign evaluate campaigns/heat-v1 --split test
```

Le manifeste du modèle contient les identifiants et hashes de tous les échantillons utilisés, le
hash logique du contenu `.npz` et une clé de reproductibilité indépendante des timestamps ZIP.
Les rapports d’évaluation utilisent uniquement les résultats nominaux déjà verrouillés et refusent
un split incomplet, sauf demande explicite avec `--allow-partial`.

## Règles pratiques

1. Versionner le triplet spécification/lock/cas avant les calculs longs.
2. Ne jamais modifier manuellement `campaign.lock.json` ou `cases.jsonl`.
3. Utiliser un nouveau seed ou une nouvelle famille plutôt que remplacer des cas existants.
4. Garder validation et test hors du split d’entraînement.
5. Archiver ensemble le modèle, son manifeste, le rapport et le dataset de référence.
