# 6. Annotation : construire la vérité terrain

Sans vérité terrain, pas de score ; sans score, pas d'optimisation. Ce chapitre explique
comment les annotations sont produites, stockées et utilisées.

## Le store : une seule source de vérité

Toutes les annotations sont **ajoutées** (jamais modifiées ni supprimées) dans
`data/annotations/round_<N>.jsonl`, un fichier par round, versionné dans git. Code :
`src/vex/annotation/store.py`.

Un enregistrement ressemble à ceci :

```json
{
  "record_id": "9f2c...",
  "doc_id": "b6bf59d4ecd0...",
  "round": 1,
  "status": "reviewed",
  "values": {"assureur": "Assur'Mirage", "immatriculation": "CL-068-QD", "garanties": null, ...},
  "source": "human",
  "annotator": "local:antoine",
  "review_seconds": 94.0,
  "comment": "deux véhicules, j'ai gardé le premier",
  "candidate_hash": null,
  "model_id": null,
  "backend": "local",
  "created_at": "2026-10-18T09:12:44+00:00",
  "reviewed_at": "2026-10-18T09:12:44+00:00"
}
```

### Les trois statuts

| Statut | Écrit par | Compte comme vérité terrain ? |
|---|---|---|
| `pre_annotated` | un modèle : la proposition montrée à l'annotateur | **non, jamais** |
| `reviewed` | un humain qui a relu et corrigé une proposition | oui |
| `blind` | un humain qui a annoté sans aucune proposition | oui |

**La vérité terrain d'un document est son enregistrement `reviewed` ou `blind` le plus
récent** (round le plus élevé, puis date la plus récente). Une proposition de modèle n'est
jamais de la vérité terrain, même si elle est plus récente.

Pour corriger une annotation fausse, on n'édite pas le fichier : on ajoute un nouvel
enregistrement `reviewed`, qui remplace l'ancien. L'historique reste lisible.

Deux protections :

- les valeurs sont validées contre le schéma à l'écriture (une date non ISO est refusée) ;
- un enregistrement dont l'annotateur serait `synthetic_meta` est refusé : les valeurs du
  générateur ne peuvent pas se déguiser en annotation.

### Pourquoi garder la proposition *et* la correction ?

Comparer les deux donne de l'information précieuse : quels champs les humains corrigent le
plus (indicateur `fields_corrected_share` du chapitre 8), et ce que le modèle de réécriture
doit apprendre (le bootstrap, plus bas).

## Les backends : comment les documents vont aux annotateurs

Un backend fait deux choses : `push` (envoyer un lot) et `pull` (récupérer le travail
terminé). Il est choisi par `annotation.backend`. Le store reste la seule source de vérité :
les backends ne font que transporter.

### Le backend local (sur un portable)

`push` écrit un fichier JSON éditable par document dans `data/inbox/round_<N>/` (dossier non
versionné) :

```json
{
  "_instructions": "Open `document`, correct `values` ...",
  "_done": false,
  "doc_id": "b6bf59d4ecd0...",
  "document": "data/raw/synthetic/syn_0010_memo.jpg",
  "status": "pre_annotated",
  "values": { "...prérempli avec la proposition du modèle..." },
  "comment": "",
  "annotator": "",
  "review_seconds": null
}
```

Pour annoter :

1. ouvrir l'image indiquée dans `document` ;
2. corriger `values` pour qu'elles correspondent **exactement** au document (`null` si le
   champ est absent ou illisible, jamais `""`). Les dates peuvent être tapées en ISO ou à la
   française (« 01/02/2025 », « 1er février 2025 ») : elles seront converties ;
3. écrire tout doute dans `comment` (« deux véhicules », « date à moitié effacée ») ;
4. mettre son nom dans `annotator`, et éventuellement le temps passé dans `review_seconds` ;
5. passer `_done` à `true`.

`pull` importe uniquement les fichiers marqués `_done: true`, puis les renomme en
`*.imported.json` pour ne jamais les importer deux fois. Un fichier dont une valeur ne passe
pas la validation (une date incompréhensible par exemple) est signalé dans le log et laissé
en place pour correction.

### Le backend Kili (au bureau)

[Kili](https://kili-technology.com) est la plateforme d'annotation de l'entreprise. Le
backend (`kili_backend.py`, SDK Kili 2.142.1) suit les modèles du dépôt `template-kili` :

- le projet a un job de transcription (texte libre) par champ, non obligatoire (vide =
  absent = `null`), plus un job `COMMENTAIRE`. La consigne de chaque job est la description
  française du champ ;
- les documents sont envoyés en PDF (les images sont converties en PDF d'une page), avec
  le `doc_id` comme identifiant externe et des métadonnées `status` et `round` ;
- les propositions du modèle sont envoyées comme pré-annotations, **sauf** pour les
  documents à l'aveugle ;
- à l'import, les labels deviennent des enregistrements du store, avec le temps de
  relecture fourni par Kili (`secondsToLabel`).

> Ce backend **n'a jamais été exécuté contre une vraie instance Kili** : Kili n'est pas
> accessible depuis le Mac de développement. Les formats sont vérifiés hors ligne avec le
> parseur officiel du SDK, et les appels sont testés contre un faux client. Lisez
> [`docs/kili_uncertainties.md`](../kili_uncertainties.md) avant le premier essai.

## Round 0 : le test, annoté à l'aveugle

```bash
uv run vex annotate push-blind
#   ... annoter ...
uv run vex annotate pull --round 0
```

Les documents de `heldout_blind` et `heldout_standin` sont envoyés **sans aucune
proposition** (tous les champs vides). Pourquoi ? Si l'annotateur voyait la proposition du
modèle, il aurait tendance à la valider même fausse (*biais d'ancrage*), et le score de test
serait trop optimiste. Annoter à l'aveugle prend plus de temps, mais c'est le prix d'une
mesure honnête.

## Rounds 1-3 : le bootstrap

Le but du bootstrap est d'obtenir rapidement une première vérité terrain **et** un meilleur
point de départ pour GEPA, sur des lots de 5, puis 10, puis 15 documents
(`bootstrap.batch_sizes`).

`vex bootstrap --round N` est une seule commande qui se souvient où on en est. On la lance
plusieurs fois :

**1er appel : sélection, pré-annotation, envoi.**

- choix de N documents dans les seaux `gepa_train` et `gepa_pareto`, pas encore annotés,
  documents réels d'abord, jamais de document de test ;
- pré-annotation par Haiku avec le prompt courant (le seed au round 1, puis le prompt
  réécrit au round précédent) ;
- écriture des enregistrements `pre_annotated` et envoi au backend.

**Appels suivants : suivi.** La commande importe les corrections disponibles et affiche
l'avancement (« 3/5 done »).

**Quand tout est relu : proposition de réécriture.**

- pour chaque document, on compare proposition et correction (avec la normalisation du
  chapitre 5) et on garde les champs réellement changés, avec le commentaire de l'annotateur ;
- le modèle `reflection` reçoit les descriptions actuelles et ces corrections, et propose de
  nouvelles descriptions **pour les champs mal compris uniquement**, avec une justification ;
- la commande affiche le diff et demande confirmation :

```
--- field:garanties (current)
+++ field:garanties (proposed)
...
- garanties: assistance phone read as guarantee
Apply this rewrite? [y/N]
```

Si vous acceptez, le nouveau prompt est enregistré dans `prompts/bootstrap-r<N>.<hash>.json`
et servira au round suivant. Vous gardez la main : rien n'est appliqué sans votre accord
(`--yes` pour accepter sans question).

Deux garanties de conception : aucun document de test ne peut entrer dans le bootstrap (le
code lève une erreur), et le modèle de réécriture ne voit ni identifiant de document ni
image, seulement des valeurs et des commentaires.

## Évaluer un prompt sur les documents relus

```bash
uv run vex eval --split gepa_pareto --candidate prompts/bootstrap-r3.ab12cd34.json
```

La commande note le prompt sur les documents relus du seau choisi et affiche le tableau du
chapitre 5. Elle refuse `heldout_blind` et `heldout_standin` : le test n'est noté que par
`vex optimize` et `vex report`.

## Les annotations suspectes

Pendant l'optimisation (chapitre 7), un modèle vérifie les erreurs en cherchant la bonne
valeur dans le document. S'il ne la trouve pas, c'est peut-être **l'annotation** qui est
fausse. Le cas est alors écrit dans `data/review_queue/suspect_annotations.jsonl` et exclu de
l'apprentissage. Relisez régulièrement ce fichier : une étiquette fausse apprendrait au
prompt une règle fausse.

Chapitre suivant : [GEPA](07-gepa.md).
