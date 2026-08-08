# Campagnes massives, reproductibles et reprenables

## Profil thermique large

Le profil `heat-wide` génère 760 cas verrouillés : 600 cas d'entraînement, 80 cas de validation
et 80 cas de test. Les familles couvrent le domaine global, les quatre coins, les fortes
diffusivités et les temps longs. Les splits restent disjoints et utilisent des plans Latin
Hypercube déterministes.

Créer puis verrouiller une nouvelle campagne :

```powershell
$env:PYTHONPATH = "src"
python -m shardsim.campaign init campaigns/heat-wide-v1 `
  --name heat-wide-v1 --seed 20260715 --profile heat-wide
python -m shardsim.campaign lock campaigns/heat-wide-v1
```

Faire un contrôle sans lancer de calcul :

```powershell
python -m shardsim.campaign status campaigns/heat-wide-v1
python -m shardsim.campaign run campaigns/heat-wide-v1 --split train --limit 3 --dry-run
```

## Boucle automatique

Lancer toute la boucle dans le terminal :

```powershell
python -m shardsim.campaign full campaigns/heat-wide-v1 `
  --cases-per-batch 50 --epochs 150 --batch-size 8 --width 16 --device cpu
```

La commande calcule d'abord la validation, puis répète : 50 références OpenFOAM, entraînement
cumulatif du même CNN, évaluation sur validation et publication du dashboard. Une relance de la
même commande reprend les cas absents et le dernier checkpoint. Le test n'est jamais consulté par
défaut. Ajouter `--test-at-end` uniquement lorsque les choix de modèle sont terminés.

Pour lancer la même commande plusieurs heures en arrière-plan sous Windows :

```powershell
.\scripts\Start-ShardSimCampaign.ps1 `
  -Campaign campaigns\heat-wide-v1 -CasesPerBatch 50 -Epochs 150 -Device cpu
```

Le script affiche les chemins des journaux. La progression peut aussi être contrôlée à tout moment :

```powershell
$env:PYTHONPATH = "src"
python -m shardsim.campaign status campaigns/heat-wide-v1
python -m shardsim.campaign dashboard campaigns/heat-wide-v1 --open
```

Pour une fenêtre de calcul limitée, ajouter `--max-batches 2` à la commande `full`, puis la
relancer plus tard sans cette option.

## Stratégie scientifique

- Utiliser une campagne et une lignée de modèles par équation, géométrie, type de condition limite
  et résolution nominale.
- Élargir d'abord les familles qui ont la plus forte erreur de validation ou le plus fort score OOD.
- Ne jamais réinjecter les cas `test` dans l'entraînement ; créer une nouvelle famille `train` avec
  un nouveau seed pour la campagne suivante.
- Comparer plusieurs seeds dans des campagnes séparées avant de promouvoir un modèle.
- Conserver 10 à 15 % des cas en validation et environ 10 % en test.
- Préférer le CPU pour la reproductibilité inter-machine ; utiliser CUDA pour le débit en acceptant
  que la reproductibilité bit à bit dépende du GPU, du pilote et de la version de PyTorch.

## Diversification des usages

Le profil large reste limité à l'équation thermique 2D, à une géométrie rectangulaire et à des
conditions de Dirichlet homogènes. Les étapes suivantes doivent créer des contrats séparés :

1. conditions limites non nulles et variables ;
2. sources thermiques internes et dépendantes du temps ;
3. géométries masquées ou maillages non structurés ;
4. transport-diffusion, puis écoulements laminaires OpenFOAM ;
5. bibliothèque d'experts avec un sélecteur fondé sur le domaine de validité et le score OOD.

Un modèle ne doit être fusionné avec un autre que si leurs sorties partagent le même schéma,
les mêmes unités, la même localisation de champ et une calibration d'incertitude comparable.
