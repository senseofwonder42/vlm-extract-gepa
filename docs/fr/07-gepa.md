# 7. GEPA : optimiser le prompt automatiquement

GEPA (bibliothèque `gepa`, version 0.1.4) améliore les **textes** d'un système : ici, les
composants de notre prompt. Pour bien le comprendre, une analogie aide.

## L'intuition

Imaginez un rédacteur de consignes qui travaille ainsi :

1. il donne ses consignes à un employé (le modèle d'extraction) ;
2. il regarde les erreurs sur quelques documents ;
3. pour chaque erreur, un collègue (le modèle de diagnostic) va voir le document et dit
   « la bonne valeur est là, en haut à droite, et l'employé l'a confondue avec autre
   chose » ;
4. le rédacteur (le modèle de réflexion) réécrit **une** consigne en tenant compte de ces
   explications ;
5. il ne garde la nouvelle consigne que si l'employé fait mieux sur les mêmes documents ;
6. il garde **plusieurs** versions de consignes, pas seulement la meilleure en moyenne :
   une version qui est la seule à bien traiter les cartes vertes mérite d'être conservée.

C'est exactement GEPA : une boucle *exécuter → lire les erreurs → réécrire → vérifier*, avec
une population de versions (un « front de Pareto ») plutôt qu'un seul champion.

Différence avec du fine-tuning : on ne touche jamais aux poids du modèle. On ne change que
le texte du prompt, ce qui est bon marché, lisible et réversible.

## Une itération en détail

```
1. choisir un parent dans le front de Pareto
2. l'évaluer sur un MINIBATCH de documents d'entraînement (avec les traces)
3. choisir le composant à réécrire                         (ErrorDrivenSelector)
4. construire le jeu réflexif : erreurs + diagnostics      (make_reflective_dataset)
5. le modèle de réflexion écrit un nouveau texte           (proposeur par défaut de GEPA)
6. évaluer l'enfant (parent + nouveau texte) sur le MÊME minibatch
      somme des scores de l'enfant > celle du parent ?
         non → rejeté
         oui → 7. évaluer l'enfant sur tout le VALSET et l'ajouter à la population
```

### 1. Choisir un parent

Avec `candidate_selection_strategy: pareto`, GEPA tire au sort un candidat parmi ceux qui
sont les meilleurs sur **au moins un** document du valset. C'est le front de Pareto.

### 2. Évaluer sur un minibatch

GEPA prend `reflection_minibatch_size` documents (3 par défaut) dans le trainset et appelle
`VexAdapter.evaluate(batch, candidate, capture_traces=True)`. Notre adaptateur :

- lance l'extracteur `extractor_eval` (température 0) sur chaque document ;
- calcule le score de chaque document avec la métrique du chapitre 5 ;
- renvoie les scores et des **trajectoires** : pour chaque document, la réponse brute, les
  valeurs lues et la ligne de diff de chaque champ.

GEPA ne lit jamais les trajectoires : il nous les rend à l'étape 4. Si un document échoue
(erreur réseau, réponse illisible), l'adaptateur ne lève pas d'exception : le document vaut
0 et la trajectoire explique pourquoi. C'est le contrat de GEPA.

Si **tous** les scores du minibatch sont parfaits, GEPA saute le reste de l'itération
(`skip_perfect_score`) : il n'y a rien à corriger. Mais l'évaluation a déjà été payée.

### 3. Choisir le composant à réécrire

GEPA demande à un « sélecteur de modules » quels composants réécrire. Le défaut de GEPA,
`round_robin`, passe en revue les 11 composants à tour de rôle, **y compris ceux qui
marchent déjà**. On gaspillerait alors des évaluations à tester une réécriture inutile.

Notre `ErrorDrivenSelector` choisit plutôt :

- `instructions` si une réponse était illisible (problème général) ;
- sinon le champ dont l'erreur pondérée sur le minibatch est la plus grande :
  Σ poids × (1 − score du champ). En cas d'égalité, l'ordre du schéma départage.

On peut revenir au comportement de GEPA avec `gepa.module_selector: round_robin`.

### 4. Construire le jeu réflexif

`VexAdapter.make_reflective_dataset` transforme les trajectoires en courtes fiches que le
modèle de réflexion va lire. Pour chaque champ en erreur du composant choisi, on appelle le
modèle `diagnostic` (Haiku) avec l'image du document, la valeur extraite et la valeur
attendue. Il doit répondre :

```json
{"located": true,
 "quote": "Virginie Launay-Lamy",
 "location": "en haut à droite du bandeau sombre",
 "explanation": "le nom n'a pas d'étiquette « Souscripteur » ; le modèle ne l'a pas associé au titulaire"}
```

Trois règles :

- **la citation est obligatoire** : un diagnostic sans `quote` ou sans `location` est jeté ;
- si le diagnostic **ne trouve pas** la valeur attendue (`located: false`), c'est peut-être
  l'annotation qui est fausse. Le cas va dans `suspect_annotations.jsonl` et n'est **pas**
  utilisé pour réécrire le prompt ;
- les diagnostics sont mis en cache par (document, champ, prédiction, gold, modèle,
  version du prompt de diagnostic) : une même erreur n'est diagnostiquée qu'une fois.

Chaque fiche suit le format recommandé par GEPA :

```
Inputs:            {field: nom_souscripteur, document: syn_0010_memo.jpg}
Generated Outputs: null
Feedback:          [omission (oubli)] nom_souscripteur: attendu/expected 'Virginie Launay-Lamy' — obtenu/got null
                   Dans le document — en haut à droite du bandeau sombre : « Virginie Launay-Lamy »
                   Cause : le nom n'a pas d'étiquette ...
```

Le composant `instructions` reçoit un résumé par document : score, erreurs, réponses
illisibles, réparations.

### 5. Réfléchir et réécrire

GEPA place le texte actuel du composant et les fiches dans un modèle de prompt fixe
(« I provided an assistant with the following instructions... ») et appelle notre
`reflection_lm`, c'est-à-dire Haiku. Le nouveau texte est lu entre ```` ``` ```` dans la
réponse. Notre classe `ReflectionLM` expose aussi un attribut `total_cost` : GEPA s'en sert
pour respecter `gepa.max_reflection_cost_usd`.

### 6. Le test d'acceptation

L'enfant est évalué sur **le même** minibatch. Il est accepté seulement si la **somme** de ses
scores est **strictement** supérieure à celle du parent. Exemple réel (notebook) :
parent 2,857, enfant 3,0 → accepté.

### 7. Le valset et le front de Pareto

Un enfant accepté est évalué sur **tous** les documents du valset (`gepa_pareto`). Ses scores
mettent à jour le front de Pareto, et le candidat de meilleure **moyenne** devient le
`best_candidate`.

### La fusion (merge)

Avec `gepa.use_merge: true`, juste après une itération qui a produit un nouveau candidat,
GEPA peut tenter une fusion au lieu d'une réécriture. Il prend deux candidats qui ont un
ancêtre commun et part d'une copie de cet ancêtre. Pour chaque composant : si un seul des
deux parents l'a modifié, il prend sa version ; si les deux l'ont modifié, il prend celle du
parent de meilleure moyenne. La fusion n'est gardée que si elle fait au moins aussi bien que
les deux parents sur un petit échantillon du valset. Elle ne coûte pas d'appel de
réflexion, et on en fait au plus `max_merge_invocations` (5).

## Le budget : `max_metric_calls`

Un **metric call** = un document évalué avec une version du prompt = un appel à
l'extracteur (sauf si le cache disque a déjà la réponse). C'est l'unité de budget de GEPA.

| Étape | Metric calls |
|---|---|
| départ : le seed sur tout le valset | \|valset\| |
| par itération : parent sur le minibatch | m |
| par itération : enfant sur le minibatch | m (sauté si le parent était parfait) |
| par itération acceptée : enfant sur le valset | \|valset\| |

Nombre d'itérations ≈ (max_metric_calls − |valset|) / (2m + p × |valset|), où p est le taux
d'acceptation. GEPA vérifie le budget **entre** les itérations : la dernière peut le
dépasser un peu (le notebook a utilisé 20 appels pour un budget de 18).

Exemple : `max_metric_calls: 300`, valset de 20 documents, m = 3, 30 % d'acceptation →
environ 23 itérations. Coût : 300 × 0,0016 $ ≈ 0,48 $ d'extraction, plus au plus 0,15 $ de
diagnostics, environ 0,05 $ de réflexion, et la notation finale sur le test. Total : **environ
0,70 à 0,90 $ par exécution**.

Deux garde-fous :

- `budget.max_usd` pour toute l'exécution, dont `budget.heldout_reserve_usd` (0,5 $) est
  réservé : GEPA s'arrête à `max_usd − réserve`, pour que la notation sur le test ait
  toujours de quoi tourner ;
- `gepa.max_reflection_cost_usd` pour les seuls appels de réflexion.

## Lancer GEPA

```bash
uv run vex optimize                          # part du dernier prompt bootstrap-r*, sinon du seed
uv run vex optimize --max-metric-calls 100   # budget réduit
uv run vex optimize --resume runs/<dossier>  # reprendre une exécution interrompue
```

`vex optimize` (code : `src/vex/optimize.py`) :

1. construit le trainset (`gepa_train` ∩ relus) et le valset (`gepa_pareto` ∩ relus), et
   vérifie qu'aucun document de test n'y figure ;
2. appelle `gepa.optimize(...)` avec l'adaptateur, le sélecteur, `ReflectionLM`, le garde-fou
   de budget et un enregistreur d'événements ;
3. enregistre le meilleur prompt dans `prompts/gepa-best.<hash>.json` ;
4. **seulement à la fin**, note le seed, le meilleur prompt et les meilleurs candidats du
   front sur `heldout_blind` et `heldout_standin`.

Pour arrêter proprement une exécution en cours, créez un fichier `gepa.stop` dans le dossier
`gepa/` du run.

## Lire les résultats

| Fichier du run | Ce qu'on y regarde |
|---|---|
| `iterations.jsonl` | événements `accepted` / `rejected` avec les scores du minibatch ; lignes `end` avec le total de metric calls |
| `proposals.jsonl` | chaque texte proposé par la réflexion, même rejeté |
| `candidates.jsonl` | chaque candidat noté sur le valset : moyenne, parents, texte complet |
| `candidates/<i>.json`, `pareto_front.json` | la population finale ; quel candidat est le meilleur sur quel document |
| `heldout.json` | seed contre meilleur prompt contre front, sur le test |
| `cost.json` | dollars et tokens par rôle |
| `gepa/run_log.txt`, `gepa/gepa_state.bin` | le log de GEPA ; l'état qui permet `--resume` |

### Ce que le notebook a montré

Le notebook [`gepa_walkthrough.ipynb`](../../notebooks/gepa_walkthrough.ipynb) déroule deux
itérations sur 5 documents. Trois enseignements, valables pour les vrais runs :

1. **Une vraie amélioration** : sur un Mémo, l'extracteur oubliait le nom du souscripteur
   (écrit sans étiquette dans l'en-tête). GEPA a réécrit la description du champ, et le
   document est passé de 0,857 à 1,0.
2. **Un valset trop facile ne départage rien** : le seed avait déjà 1,0 sur les 2 documents
   du valset, l'enfant aussi. En cas d'égalité, GEPA garde le premier : le « meilleur »
   prompt est resté le seed. Il faut un valset assez grand et assez difficile.
3. **Un trainset parfait gaspille le budget** : après la première itération, le
   trainset était parfait. Les itérations 2 à 5 ont été sautées, mais chacune a coûté
   3 metric calls. La bonne réponse n'est pas plus d'itérations mais **plus de documents
   difficiles annotés** (chapitre 8).

### Signaux à surveiller

- beaucoup de lignes `end` avec `proposal_accepted: false` et aucun `rejected` : GEPA saute
  des minibatches parfaits ; annotez des documents plus durs ;
- des enfants acceptés qui font tous comme le seed sur le valset : valset trop petit ou trop
  facile ;
- beaucoup de rejets de suite : les fiches de réflexion sont peut-être trop pauvres
  (diagnostics jetés) ; regardez `proposals.jsonl` ;
- `suspect_annotations.jsonl` qui grossit : corrigez les étiquettes avant d'optimiser.

## Correspondance concept GEPA → code

| Concept | Code |
|---|---|
| candidat, composants | `vex.prompt.seed_candidate`, `render` |
| adaptateur (`GEPAAdapter`) | `vex.gepa_adapter.VexAdapter` |
| donnée (`DataInst`) | `vex.gepa_adapter.DocInstance` (doc_id, chemin, gold) |
| `evaluate` | `VexAdapter.evaluate` → `extract_document` + `score_document` |
| trajectoire | dictionnaire construit par `VexAdapter._trajectory` |
| `make_reflective_dataset` | `VexAdapter.make_reflective_dataset` + `vex.diagnostic.Diagnostician` |
| sélecteur de modules | `vex.gepa_adapter.ErrorDrivenSelector` |
| `reflection_lm` | `vex.optimize.ReflectionLM` |
| trainset / valset | `effective_split("gepa_train" / "gepa_pareto")` |
| conditions d'arrêt | `vex.optimize.BudgetStopper`, fichier `gepa.stop`, `max_metric_calls` |
| callbacks | `vex.optimize.RunRecorder` (fichiers `*.jsonl`) |
| appel principal | `vex.optimize.run_optimize` |
| notation sur le test (hors GEPA) | `vex.optimize.score_heldout` |

## Pistes d'extension (non implémentées)

- **Images dans la réflexion** : le proposeur par défaut de GEPA 0.1.4 sait déjà envoyer des
  images au modèle de réflexion. `gepa.reflection_images: true` ajoute la première page à
  chaque fiche. Désactivé par défaut : le diagnostic regarde déjà l'image et cite la zone,
  et les images multiplient le coût de la réflexion.
- **Proposeur multimodal personnalisé** : en assignant une fonction à
  `VexAdapter.propose_new_texts`, GEPA l'appellerait à la place de son proposeur par défaut,
  par exemple pour réécrire plusieurs composants liés en même temps.

Pour aller plus loin : [`docs/gepa_guide.md`](../gepa_guide.md) (en anglais).

Chapitre suivant : [Apprentissage actif et suivi](08-apprentissage-actif-et-suivi.md).
