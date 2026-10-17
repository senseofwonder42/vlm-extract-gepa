# 5. La métrique

La métrique dit si une extraction est bonne. Tout le reste en dépend : GEPA optimise ce
score, l'apprentissage actif s'en sert, et le suivi des rounds le rapporte. Une métrique
mal conçue ferait optimiser la mauvaise chose ; ce chapitre détaille donc chaque choix.

Code : `src/vex/metric/normalize.py`, `compare.py` et `report.py`.

## Étape 1 : normaliser avant de comparer

Comparer des chaînes brutes serait trop sévère : « Citroën » et « CITROEN » désignent la
même marque, « AB-123-CD » et « ab 123 cd » la même plaque. On passe donc les deux côtés
(vérité terrain et prédiction) par le **même** normaliseur, choisi selon le type du champ :

| Type | Champs | Normalisation | Exemple |
|---|---|---|---|
| texte | `assureur`, `nom_souscripteur`, `adresse_souscripteur`, `marque_vehicule` | minuscules, sans accents ni ligatures, ponctuation remplacée par des espaces, espaces fusionnés | « Citroën » → `citroen` |
| identifiant | `numero_contrat`, `code_assureur` | lettres et chiffres seulement, en majuscules | « AF-77-31-552 » → `AF7731552` |
| date | `date_debut_validite`, `date_fin_validite` | analyse en date, sortie ISO | « 1er février 2025 » → `2025-02-01` |
| plaque | `immatriculation` | forme canonique SIV ou FNI | « ab 123 cd » → `AB-123-CD` |
| liste | `garanties` | ensemble de textes normalisés | ordre et doublons ignorés |

Quelques détails qui comptent :

- **Dates** : on accepte `01/02/2025`, `1/2/25`, `01.02.2025`, `01-02-2025`, `01 02 2025`,
  `1er février 2025`, `3 déc. 2025` et l'ISO. Le jour vient **en premier** dans les formats
  numériques français. Années à deux chiffres : 00-69 → 20xx, 70-99 → 19xx. Une date
  impossible (31/02/2025) n'est pas convertie : elle reste du texte et ne pourra jamais
  être égale à une vraie date.
- **Identifiants** : les zéros de tête sont gardés. « 0457 » et « 457 » sont deux codes
  assureur différents.
- **Plaques** : un VIN (17 caractères) reste tel quel ; il ne deviendra jamais une plaque.
- **Montants** (`1 234,56 €` → `1234.56`) : le normaliseur existe et il est testé, mais aucun
  champ du schéma n'est un montant pour l'instant.

## Étape 2 : comparer champ par champ

Pour chaque champ, après normalisation :

1. **Les deux sont vides** → correct (bien vu que le champ est absent) ;
2. **Gold rempli, prédiction vide** → **omission** (le modèle a raté le champ) ;
3. **Gold vide, prédiction remplie** → **hallucination** (le modèle a inventé une valeur) ;
4. **Les deux remplis** : comparaison selon le champ (voir ci-dessous). Égal → **correct**,
   différent → **valeur erronée** (`wrong_value`).

Distinguer ces trois types d'erreur est utile : une omission demande souvent de dire où
chercher, une hallucination de dire quoi exclure.

### Trois façons de comparer

- **Égalité exacte** (après normalisation), par défaut.
- **Correspondance approximative** pour les champs listés dans `metric.fuzzy_fields` :
  `assureur`, `nom_souscripteur`, `adresse_souscripteur`. Une adresse transcrite avec une
  virgule en moins n'est pas une vraie erreur. On calcule deux similarités (de 0 à 100)
  avec `rapidfuzz` et on garde la meilleure :
  - `token_sort_ratio`, insensible à l'ordre des mots (« Leroux Jean-Baptiste » =
    « Jean-Baptiste Leroux ») ;
  - `ratio` sur les textes **sans espaces**, insensible à un espace en trop. Ce second calcul
    a été ajouté après un vrai cas : « Pottiernec » contre « Pottier nec » ne faisait que
    76 avec le premier.

  Le champ est correct si la meilleure similarité atteint `metric.fuzzy_threshold` (90).
  Le calcul est déterministe : la même paire donne toujours le même verdict.
- **F1 sur les ensembles** pour `garanties`, qui donne un crédit partiel :

  ```
  F1 = 2 × |gold ∩ prédiction| / (|gold| + |prédiction|)
  ```

  Gold `{Vol, Incendie}`, prédiction `{Vol}` → F1 = 2×1/(2+1) ≈ 0,67. Le champ compte alors
  comme `wrong_value` avec un score de 0,67.

## Étape 3 : le score du document

Le score d'un document est la **moyenne pondérée** des scores de ses champs. Les poids
(`metric.weights`) reflètent l'importance métier :

| Champ | Poids | | Champ | Poids |
|---|---|---|---|---|
| `immatriculation` | 3 | | `assureur` | 1 |
| `numero_contrat` | 2 | | `nom_souscripteur` | 1 |
| `date_debut_validite` | 2 | | `code_assureur` | 1 |
| `date_fin_validite` | 2 | | `garanties` | 1 |
| `adresse_souscripteur` | 0,5 | | `marque_vehicule` | 0,5 |

Le total vaut 14. Exemples :

- une plaque fausse et tout le reste juste : 1 − 3/14 ≈ **0,786** ;
- une garantie inventée et tout le reste juste : 1 − 1/14 ≈ **0,929** ;
- réponse illisible (réparation échouée) : **0**, et **tous** les champs sont typés
  `omission`, y compris ceux dont le gold est vide. On ne veut pas récompenser une absence
  de réponse.

La configuration refuse des poids qui ne couvrent pas exactement les dix champs du schéma.

## Étape 4 : le retour textuel

Chaque champ produit aussi une ligne lisible, en français et en anglais :

```
[wrong_value (valeur erronée)] immatriculation: attendu/expected 'AB-123-CD' — obtenu/got 'AB-123-CO'
[hallucination (valeur inventée)] garanties: attendu/expected null — obtenu/got ['Assistance 24H/24']
[wrong_value (valeur erronée)] garanties: attendu/expected [...] — obtenu/got [...] (F1=0.67; manquant/missing ['incendie']; en trop/extra [])
```

Ce n'est pas qu'un affichage : c'est **le texte que lit le modèle de réflexion de GEPA**
pour comprendre ce qui ne va pas (chapitre 7). Il doit donc dire précisément ce qui était
attendu, ce qui a été produit et quel type d'erreur c'est.

## Étape 5 : le rapport

Sur un ensemble de documents, `build_report` calcule le score global (moyenne des scores
de documents), le score moyen de chaque champ, le nombre d'erreurs de chaque type par champ
et le nombre de réponses illisibles. Exemple réel (`vex data sanity` sur 20 documents
synthétiques avec le prompt de départ) :

```
field                   score        correct       omission  hallucination    wrong_value
assureur                1.000             20              0              0              0
...
garanties               0.800             16              0              4              0
GLOBAL (weighted)       0.986            196              0              4              0
```

Les 4 hallucinations sont toutes le même phénomène, une vraie ambiguïté : sur les Mémos, le
modèle prend le bloc « ASSISTANCE 24H/24 » (un numéro de téléphone) pour une garantie
souscrite. C'est exactement le genre d'erreur que GEPA doit apprendre à corriger.

## Où la métrique est utilisée

| Endroit | Usage |
|---|---|
| `vex data sanity` | contrôle rapide contre les valeurs du générateur |
| `vex eval` | score d'un prompt sur les documents relus d'un seau |
| GEPA (`VexAdapter.evaluate`) | le score que GEPA maximise |
| bootstrap, `vex report` | « ce champ a-t-il été corrigé ? » (égalité après normalisation) |
| `vex select` | désaccord entre deux extractions, mesuré avec la même normalisation |

Chapitre suivant : [Annotation](06-annotation.md).
