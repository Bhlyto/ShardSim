# Consulter et réutiliser les résultats

Ce parcours est conçu pour un utilisateur qui ne modifie pas le code.

## Ouvrir le tableau de bord

Une seule commande reconstruit les fichiers concaténés et ouvre le tableau de bord :

```powershell
python -m shardsim.campaign dashboard campaigns/heat-v1 --open
```

Le tableau de bord permet de :

- filtrer les cas par split et famille ;
- comparer la pré-simulation grossière, le nominal, la correction et l’erreur absolue ;
- consulter les paramètres, erreurs et temps de calcul de chaque cas ;
- comparer les versions de modèles et leurs scores de validation/test ;
- consulter l’historique des lancements manuels.

Le fichier est généré dans `outputs/<empreinte>/reports/dashboard.html`. Il est autonome : il peut
être copié, archivé ou ouvert par double-clic, sans serveur web.

## Résultats concaténés

Le tableau de bord génère également :

- `results.csv` : tableau lisible dans Excel ou LibreOffice, avec une ligne par simulation ;
- `combined-results.npz` : dataset ML normalisé contenant l’ordre des cas, les paramètres scalaires,
  les limites, les champs grossiers, nominaux, les corrections et les cartes d’erreur ;
- `export.manifest.json` : hashes et contrat des tableaux pour vérifier leur intégrité.

Pour générer uniquement ces exports :

```powershell
python -m shardsim.campaign export campaigns/heat-v1
```

## Comparer et réutiliser les modèles

Chaque entraînement produit une version immuable dans `models/registry/<clé>/`. La clé dépend de la
spécification, des hyperparamètres et des hashes exacts des données d’entraînement. Un entraînement
identique retrouve donc la même version au lieu de créer une copie ambiguë.

Ces versions sont des **checkpoints d’une même lignée cumulative**, et non des modèles indépendants.
Pour le surrogate thermique actuel, chaque checkpoint réajuste le même algorithme sur tous les cas
d’entraînement disponibles. Le manifeste enregistre la lignée, le parent, le numéro du checkpoint,
les nouveaux cas et le nombre total de cas réutilisés.

```powershell
python -m shardsim.campaign models campaigns/heat-v1
python -m shardsim.campaign activate campaigns/heat-v1 --key <début-de-clé-affiché>
```

`activate` choisit le modèle utilisé par les prochaines évaluations. Les rapports sont conservés
par clé de modèle, ce qui permet une comparaison historique sans écraser les résultats précédents.

Après chaque checkpoint, relancer l’évaluation sur le même split de validation :

```powershell
python -m shardsim.campaign evaluate campaigns/heat-v1 --split validation
python -m shardsim.campaign dashboard campaigns/heat-v1 --open
```

Le dashboard affiche alors la courbe erreur/nombre de cas, le gain face au calcul grossier, l’erreur
sur les gradients, la couverture d’incertitude, les contraintes thermiques et l’accélération obtenue.

## À propos des CNN

ShardSim fournit désormais un petit U-Net résiduel thermique. Il apprend la correction entre le
champ grossier et le champ nominal, tout en imposant exactement les conditions de bord. PyTorch est
une dépendance optionnelle :

```powershell
python -m pip install -e ".[cnn]"
```

Le premier checkpoint nécessite au moins quatre cas d’entraînement terminés :

```powershell
python -m shardsim.campaign run campaigns/heat-v1 --split train --limit 4
python -m shardsim.campaign train campaigns/heat-v1 --allow-partial `
  --algorithm heat-residual-unet --epochs 200 --width 8 --device cpu
```

Après l’ajout de nouveaux calculs nominaux, relancer exactement la même commande. Le modèle recharge
les poids, l’optimiseur et les normalisations du checkpoint parent, puis rejoue tous les anciens et
nouveaux cas. Si aucun nouveau cas n’est disponible, aucun checkpoint artificiel n’est créé.

Le CPU est choisi par défaut pour favoriser la reproductibilité. CUDA est disponible avec
`--device cuda`, mais les résultats numériques ne sont pas garantis identiques entre CPU, GPU,
plateformes ou versions de PyTorch. Le manifeste enregistre donc la version de PyTorch, CUDA, le
device, le seed et le mode déterministe.

On peut entraîner le modèle local et le CNN sur le même dataset verrouillé, sans relancer OpenFOAM :

```powershell
python -m shardsim.campaign train campaigns/heat-v1 --algorithm heat-local-residual
python -m shardsim.campaign train campaigns/heat-v1 --allow-partial `
  --algorithm heat-residual-unet --epochs 200 --width 8 --device cpu
python -m shardsim.campaign models campaigns/heat-v1
```

Le dashboard sépare les lignées et permet ensuite de comparer leurs erreurs, gradients, couvertures
et accélérations sur le même split de validation.
