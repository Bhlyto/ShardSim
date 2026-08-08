# Contrat de scénario ShardSim V1

Le schéma V1 est volontairement strict. Les champs inconnus sont rejetés afin qu’une
faute de frappe ne change jamais silencieusement une simulation.

```json
{
  "schema_version": "1.0",
  "scenario_id": "heat-gaussian-demo",
  "model": "heat-2d",
  "parameters": {
    "alpha": 0.02,
    "t_end": 0.03,
    "extent": [1.0, 1.0]
  },
  "initial_conditions": {
    "type": "gaussian",
    "shape": [65, 65],
    "center": [0.5, 0.5],
    "sigma": [0.09, 0.09],
    "amplitude": 1.0,
    "baseline": 0.0
  },
  "boundary_conditions": {
    "top": 0.0,
    "bottom": 0.0,
    "left": 0.0,
    "right": 0.0
  },
  "solver": {
    "backend": "internal",
    "grid_shape": [65, 65],
    "safety_factor": 0.9
  },
  "seed": null,
  "metadata": {}
}
```

## Sémantique

- `schema_version` vaut exactement `1.0`.
- `scenario_id` est un identifiant ASCII de 1 à 128 caractères.
- `model` vaut exactement `heat-2d`.
- `alpha` est la diffusivité thermique positive en m²/s.
- `t_end` est l’horizon physique positif en secondes.
- `extent` contient les longueurs physiques x et y positives en mètres.
- La condition initiale V1 est une gaussienne paramétrée sur une grille d’au moins 3×3.
- Les quatre frontières sont des conditions de Dirichlet en kelvins.
- Le backend V1 est `internal`; `grid_shape` décrit la grille de sortie.
- `safety_factor` appartient à l’intervalle ]0, 1].
- `seed` est un entier ou `null`. Le modèle V1 est déterministe et n’utilise donc pas
  la seed, mais celle-ci reste enregistrée pour l’identité du scénario.

Les valeurs par défaut normalisées sont `extent=[1,1]`, `center=[0.5,0.5]`,
`sigma=[0.1,0.1]`, `amplitude=1`, `baseline=0` et `safety_factor=0.9`.

## Contrat de résultat

`result.json` utilise `result_schema_version: "1.0"` et contient au minimum :

- l’identifiant, le statut et le code de sortie ;
- les instants de début et de fin et la durée du calcul ;
- les versions ShardSim et du modèle physique ;
- la seed et le hash canonique du scénario ;
- la discrétisation effectivement utilisée ;
- des métriques physiques et numériques ;
- l’environnement utile ;
- la liste des artifacts, leurs tailles et leurs SHA-256 ;
- une `reproducibility_key` stable si le scénario, le modèle et le champ produit sont identiques.

`shardsim inspect` recalcule les checksums et rejette tout artifact absent, altéré ou
référencé hors du répertoire du résultat.

## Codes de sortie

- `0` : validation, exécution ou inspection réussie ;
- `1` : le scénario était valide mais le calcul physique a échoué ;
- `2` : entrée invalide, résultat invalide ou répertoire de sortie déjà occupé.
