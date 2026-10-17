# 9. Guide pratique

Ce chapitre est une recette : un cycle complet pas à pas, puis les problèmes fréquents et
comment modifier le projet sans rien casser.

## Un cycle complet, de zéro

### 0. Vérifier l'installation (5 minutes, ~0,01 $)

```bash
uv sync
uv run vex doctor                 # clé présente, fournisseurs compatibles
uv run pytest                     # ~460 tests, tous hors ligne
uv run pytest -m live             # 1 vrai appel à l'extracteur
```

### 1. Construire les données (une seule fois)

```bash
uv run vex data fetch-specimens
uv run vex data synth
uv run vex data register
uv run vex data split
```

Si `data/registry.jsonl` et `data/splits.yaml` existent déjà (ils sont versionnés), il suffit
de régénérer les fichiers bruts : `synth` reproduit exactement les mêmes octets, donc les
mêmes `doc_id`.

### 2. Annoter le jeu de test à l'aveugle (round 0)

```bash
uv run vex annotate push-blind
```

Sur le Mac : 15 fichiers apparaissent dans `data/inbox/round_0/`, tous les champs à `null`.
Pour chacun, ouvrez l'image, remplissez `values`, mettez `_done: true`. Puis :

```bash
uv run vex annotate pull --round 0
```

Comptez 2 à 4 minutes par document. C'est l'étape la plus importante pour la fiabilité de
toutes les mesures qui suivent : prenez votre temps.

### 3. Bootstrap (rounds 1 à 3)

```bash
uv run vex bootstrap --round 1    # 5 documents pré-annotés par Haiku (~0,005 $)
#   corriger les fichiers de data/inbox/round_1/
uv run vex bootstrap --round 1    # import + proposition de réécriture → accepter ou non
uv run vex report --round 1
```

Puis la même chose avec `--round 2` (10 documents) et `--round 3` (15 documents). À la fin,
environ 30 documents relus et un prompt `prompts/bootstrap-r3.<hash>.json`.

### 4. Optimiser avec GEPA (~0,70 à 0,90 $)

```bash
uv run vex optimize
```

Regardez `heldout.json` dans le dossier du run : le meilleur prompt fait-il mieux que le seed
sur le test ? Lisez aussi `iterations.jsonl` (chapitre 7).

### 5. Boucle d'apprentissage actif (rounds 4 et suivants)

```bash
uv run vex select                 # 20 documents choisis, pré-annotés
#   corriger data/inbox/round_4/
uv run vex annotate pull --round 4
uv run vex optimize
uv run vex report
```

Recommencez jusqu'à ce que `vex report` signale un plateau.

## Dépannage

| Symptôme | Cause probable | Que faire |
|---|---|---|
| `OPENROUTER_API_KEY is missing` | pas de `.env` ou clé vide | `cp .env.example .env` et renseigner la clé |
| erreur 404 « No endpoints found » | un paramètre n'est pas accepté par le fournisseur épinglé (c'était `min_p`) | `uv run vex doctor` indique lequel ; retirez-le ou changez de fournisseur |
| `Extra inputs are not permitted` | faute de frappe dans le YAML ou un `--set` | corriger la clé ; `vex config show` pour vérifier |
| `Budget reached` | `budget.max_usd` atteint | augmenter avec `--set budget.max_usd=...` ou réduire `max_metric_calls` |
| un fichier de l'inbox n'est pas importé | `_done` est encore `false`, ou une valeur est invalide | lire le log : il nomme le fichier et l'erreur (souvent une date) |
| `No gold document in gepa_pareto yet` | aucune annotation relue dans ce seau | faire au moins un round de bootstrap |
| `heldout_blind: no gold document yet` | le round 0 n'a pas été annoté | `vex annotate push-blind`, annoter, puis `pull --round 0` |
| un relancement coûte 0 $ | c'est le cache disque | normal ; `--no-cache` pour forcer de vrais appels |
| GEPA « saute » toutes les itérations | le trainset est déjà parfait | annoter des documents plus difficiles (`vex select`) |
| les `doc_id` ne correspondent plus au registre | fichiers régénérés avec d'autres versions de bibliothèques | ne pas changer Pillow, PyMuPDF ou Faker sans régénérer registre et splits |

## Pièges à éviter

1. **Ne jamais utiliser `synthetic_meta` comme étiquette.** Ce sont les intentions du
   générateur, pas ce qu'on lit sur un scan dégradé. Seuls les tests et `vex data sanity` y
   ont droit.
2. **Ne pas montrer le test à un modèle qui écrit des prompts**, ni choisir un prompt d'après
   son score de test (chapitre 8). Le code l'interdit pour GEPA, le diagnostic et le
   bootstrap ; restez vigilant si vous écrivez de nouveaux scripts.
3. **Ne pas éditer `data/splits.yaml` ni `data/annotations/*.jsonl` à la main.** Pour corriger
   une annotation, ajoutez un nouvel enregistrement `reviewed` (il remplace l'ancien).
4. **Ne pas monter la version de Pillow.** Kili 2.142.1 impose `pillow<10.1`, et changer
   l'encodeur d'images changerait les `doc_id` des documents synthétiques.
5. **Ne pas comparer deux scores obtenus avec des modèles ou fournisseurs différents.**
   Changer de fournisseur (quantification différente) change les réponses.
6. **Lire les annotations suspectes** (`data/review_queue/suspect_annotations.jsonl`) avant
   chaque optimisation.

## Sur le Mac ou au bureau

| | Mac | Bureau |
|---|---|---|
| Annotation | backend `local`, fichiers JSON dans `data/inbox/` | backend `kili` : renseigner `KILI_API_KEY` et `KILI_API_ENDPOINT` ; le premier run crée le projet, reporter son id dans `annotation.kili.project_id` |
| Test | `heldout_standin` (15 cartes vertes synthétiques réservées) | `splits.standin_layouts: []`, ajouter des documents réels anonymisés dans `data/raw/real/`, puis `vex data register && vex data split` |
| Données personnelles | aucune (documents fictifs) | envisager `provider.data_collection: deny` dans la config de l'extracteur ; vérifier avec `vex doctor` |

Avant le premier essai au bureau, suivez dans l'ordre la liste de
[`docs/kili_uncertainties.md`](../kili_uncertainties.md) : le backend Kili n'a jamais tourné
contre une vraie instance.

## Modifier le projet

### Ajouter un champ (exemple : `franchise`)

1. `src/vex/schema.py` : ajouter le nom dans `FIELD_NAMES` et un `Field(description=...)`
   précis dans `AttestationAuto` (où chercher, quoi exclure, format, « absent → null »).
2. `src/vex/metric/compare.py` : ajouter le champ dans `FIELD_KINDS` avec son type de
   normalisation. Pour un montant, il faudra un type `amount` qui utilise le
   `normalize_amount` déjà écrit.
3. `configs/default.yaml` : ajouter son poids dans `metric.weights` (la configuration refuse
   de démarrer sinon).
4. Lancer `uv run pytest` : les tests signalent ce qui manque.

Le prompt de départ, le schéma JSON envoyé au modèle et l'interface Kili sont générés
automatiquement à partir du schéma : rien à faire de ce côté. Attention, les prompts déjà
enregistrés n'ont pas le nouveau composant : il faut repartir du seed
(`uv run vex prompt seed`) et refaire au moins un round d'annotation qui couvre ce champ.

### Changer de modèle

Modifiez `models.<rôle>.model` dans le YAML, puis lancez `uv run vex doctor` : la commande
vérifie que le nouveau modèle accepte les paramètres envoyés. Pensez au raisonnement (à
désactiver ou non selon le rôle) et au prix dans `pricing_fallback_usd_per_m`. Les scores
précédents ne sont plus comparables : refaites une évaluation de référence.

### Tester un changement sans dépenser

Tous les tests unitaires simulent OpenRouter (un faux transport HTTP dans
`tests/conftest.py`). Pour un nouveau comportement, écrivez d'abord un test avec ce faux
modèle, comme dans `tests/test_gepa.py` qui fait tourner un vrai `gepa.optimize` contre un
modèle simulé.

## Pour aller plus loin

- Exécuter le notebook [`gepa_walkthrough.ipynb`](../../notebooks/gepa_walkthrough.ipynb)
  cellule par cellule (quelques centimes, presque gratuit en relance).
- Lire `src/vex/gepa_adapter.py`, abondamment commenté.
- Consulter [`docs/PLAN.md`](../PLAN.md) pour l'historique des décisions et leurs raisons.
