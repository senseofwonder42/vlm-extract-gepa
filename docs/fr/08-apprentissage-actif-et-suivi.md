# 8. Apprentissage actif et suivi des rounds

Annoter coûte du temps humain. Après le bootstrap, on veut annoter en priorité les
documents qui **apprendront le plus** au système : ceux où le modèle hésite. C'est
l'apprentissage actif. Ensuite, on mesure chaque round pour savoir quand s'arrêter.

## `vex select` : choisir le prochain lot

```bash
uv run vex select              # round N+1, 20 documents par défaut
uv run vex select --size 10
```

Code : `src/vex/active/`.

### Étape 1 : le pool

Le pool contient les documents des seaux `gepa_train` et `gepa_pareto` qui n'ont encore
**aucune** annotation, jamais de document de test. Pour borner le coût, on en tire au hasard
au plus `active.max_pool` (60).

### Étape 2 : les candidats

On récupère les meilleurs prompts du dernier `vex optimize` (les `top_k_candidates` = 3
meilleurs sur le valset). Sans optimisation préalable, on prend le dernier prompt enregistré
(`gepa-best` ou `bootstrap-r*`), sinon le seed. Le premier est le « meilleur actuel ».

### Étape 3 : quatre signaux d'incertitude

Chaque signal donne un nombre entre 0 (le modèle est sûr) et 1 (il hésite beaucoup).

**(a) Désaccord entre candidats de Pareto.** On extrait le document avec chacun des 3
meilleurs prompts. S'ils ne donnent pas les mêmes valeurs, le document est « à la
frontière » de ce que les prompts savent faire. Calcul, pour chaque champ : la part des
extractions qui diffèrent de la valeur majoritaire (après normalisation), puis une moyenne
pondérée par les poids des champs. Un désaccord sur la plaque (poids 3) pèse donc plus
qu'un désaccord sur la marque (poids 0,5). Ce signal n'existe que s'il y a au moins deux
candidats.

**(b) Auto-cohérence.** On extrait le document N fois (3 par défaut) avec l'extracteur à
température 0,9. Si le modèle répond différemment à chaque tirage, il n'est pas sûr de lui.
On compare aussi l'extraction déterministe à celle de Haiku : deux modèles différents qui ne
sont pas d'accord, c'est un autre indice. Le signal est la moyenne des deux désaccords.

**(c) Règles métier.** Des vérifications sans modèle :

- les champs obligatoires sont présents (assureur, numéro de contrat, souscripteur,
  immatriculation, les deux dates) ;
- les dates sont valides et plausibles (entre aujourd'hui − 5 ans et aujourd'hui + 2 ans) ;
- la date de fin est postérieure à la date de début ;
- la plaque est au format SIV ou FNI.

Le signal est la part de vérifications échouées. Une réponse illisible vaut 1.

**(d) Vérificateur.** Le modèle `diagnostic` reçoit l'image et les valeurs extraites, et doit
dire pour chacune si elle figure bien dans le document, **en citant le texte**. Le signal est
la part pondérée des valeurs non trouvées. C'est le signal le plus cher, donc on ne le
calcule que sur les `verifier_shortlist` (40) documents les plus incertains selon (a)+(b)+(c).

**Combinaison.** Moyenne pondérée par `active.weights` (0,35 / 0,25 / 0,2 / 0,2) des signaux
**disponibles** : un signal non calculé est ignoré, pas compté comme 0, sinon un document
sans vérificateur paraîtrait artificiellement sûr.

### Étape 4 : composer le lot

- **Les incertains, mais variés.** Choisir seulement les plus incertains donne souvent des
  quasi-doublons (le même layout qui échoue de la même façon). On calcule donc un
  *embedding* de la première page de chaque document, on regroupe les documents en
  `n_clusters` (8) groupes avec KMeans, puis on pioche **à tour de rôle** dans chaque groupe,
  le plus incertain d'abord.
- **Un échantillon aléatoire de documents « sûrs ».** 15 % du lot
  (`random_confident_fraction`) est tiré au hasard parmi la moitié la moins incertaine.
  Pourquoi ? Les signaux peuvent se tromper : un modèle peut être sûr de lui **et** faux. Ces
  documents mesurent l'erreur résiduelle que les signaux ne voient pas.

Embeddings : avec `uv sync --extra embed`, un ResNet50 au format ONNX (94 Mo, téléchargé une
fois). Sans cette option, une vignette 32×32 en niveaux de gris de la première page, qui
suffit à distinguer des mises en page différentes. (La bibliothèque `fastembed`, prévue au
départ, est incompatible avec la version de Pillow imposée par Kili.)

### Étape 5 : envoyer

Le lot est envoyé comme round N+1 en `pre_annotated`. La pré-annotation est l'extraction de
Haiku déjà calculée pour le signal (b) : pas d'appel en plus. Le fichier
`runs/<...>-select-r<N>/selection.json` détaille, pour chaque document, ses signaux et la
raison de son choix (`uncertain` ou `random_confident`).

### Ce qu'on a observé

Sur 8 documents synthétiques (0,044 $), presque tous les signaux valaient 0 : les tirages,
Haiku, les règles et le vérificateur étaient d'accord. Le choix a donc été dicté par la
diversité seule. C'est cohérent avec le fait que le jeu synthétique est facile pour ce
modèle : **l'apprentissage actif deviendra utile avec de vrais documents**, plus variés et
plus abîmés.

Coût indicatif par document du pool : 1 extraction déterministe + (K−1) candidats + N tirages
+ 1 Haiku, plus le vérificateur pour la liste restreinte. Soit environ 0,005 à 0,01 $ par
document, donc 0,30 à 0,60 $ pour un pool de 60.

## `vex report` : mesurer chaque round

```bash
uv run vex report               # dernier round
uv run vex report --round 2
```

La commande note le prompt courant sur le test (`heldout_blind`, ou `heldout_standin` s'il n'y
a pas encore de document réel annoté) et écrit une ligne dans `reports/rounds.csv`. Relancer
la commande pour un même round **remplace** sa ligne.

| Colonne | Signification |
|---|---|
| `heldout_split`, `heldout_n` | quel jeu de test a servi, et combien de documents |
| `heldout_global`, `heldout_<champ>` | score de test, global et par champ |
| `n_gold_train`, `n_gold_pareto` | documents relus dans chaque seau |
| `fields_corrected_share` | part des champs pré-annotés que les humains ont corrigés |
| `median_review_seconds` | temps médian de relecture par document |
| `suspect_annotations` | nombre d'annotations suspectes signalées |
| `usd_round`, `usd_cumulative` | dépense du round et dépense totale |
| `plateau` | `True` si le score de test stagne |

Comment lire ces indicateurs :

- `fields_corrected_share` qui baisse au fil des rounds : les pré-annotations s'améliorent,
  les annotateurs gagnent du temps ;
- `median_review_seconds` qui baisse aussi : même conclusion, côté humain ;
- `suspect_annotations` qui monte : il faut relire des étiquettes avant de continuer.

### Quand s'arrêter ?

Le drapeau `plateau` passe à `True` quand le score de test a gagné moins de
`tracking.plateau_delta` (0,01) à **chacun** des `tracking.plateau_rounds` (2) derniers rounds.
Par exemple, des scores 0,800 → 0,805 → 0,807 donnent un plateau ; 0,80 → 0,85 → 0,855 non.

Règle importante : utilisez le score de test pour décider **quand vous arrêter**, jamais pour
**choisir entre deux prompts**. Si on choisissait le prompt qui marche le mieux sur le test,
round après round, le test deviendrait un second valset, et son score cesserait d'être une
mesure honnête.

Chapitre suivant : [Guide pratique](09-guide-pratique.md).
