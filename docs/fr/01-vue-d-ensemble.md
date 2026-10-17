# 1. Vue d'ensemble

## Le problème

Un assureur reçoit chaque jour des justificatifs d'assurance automobile : attestations,
« Mémos Véhicule Assuré » (obligatoires depuis le 1er avril 2024), cartes internationales
d'assurance (blanches depuis 2020), anciennes cartes vertes. Ce sont des PDF ou des photos,
souvent scannés de travers, flous, parfois tournés.

On veut en extraire automatiquement dix champs :

| Champ | Exemple |
|---|---|
| `assureur` | Mutuelle Imaginaire de l'Ouest |
| `numero_contrat` | MIO/AUTO/2025/00981 |
| `nom_souscripteur` | Jean-Baptiste Leroux |
| `adresse_souscripteur` | 14 bis avenue du Port, 44000 Nantes |
| `immatriculation` | AB-123-CD (format SIV) ou 1234 AB 56 (ancien format FNI) |
| `marque_vehicule` | Renault |
| `date_debut_validite` | 2025-03-01 (toujours au format ISO) |
| `date_fin_validite` | 2026-02-28 |
| `code_assureur` | 0457 |
| `garanties` | ["Responsabilité civile", "Vol", "Bris de glace"] |

Un champ absent du document vaut `null`, jamais une chaîne vide.

## La solution en une phrase

Un **modèle de vision-langage** (un VLM, ici Qwen3.8-27B) lit les pages du document et
renvoie un JSON. La qualité dépend surtout du **prompt** qu'on lui donne, et ce prompt est
amélioré automatiquement par **GEPA**, un optimiseur de prompts.

## La difficulté : on part de zéro annotation

Pour améliorer un prompt, il faut savoir quand le modèle se trompe, donc connaître les
bonnes réponses : une **vérité terrain** (ou « gold »). Au départ, on n'en a aucune.

Le projet construit donc **en même temps** :

1. la vérité terrain, par des humains qui relisent et corrigent les propositions d'un
   modèle (dans Kili au bureau, dans des fichiers JSON sur un portable) ;
2. le prompt optimisé, à partir de ces corrections.

Chaque étape est guidée par une évaluation : on mesure avant de décider.

## La boucle complète

```
                 ┌──────────────────────────────────────────────────────────┐
                 │                                                          │
  documents ──►  │  1. Données : spécimens réels + 200 synthétiques          │
                 │     découpés une fois pour toutes en splits (chap. 3)     │
                 │                                                          │
                 │  2. Round 0 : les documents de test sont annotés          │
                 │     « à l'aveugle », sans proposition du modèle (chap. 6) │
                 │                                                          │
                 │  3. Rounds 1-3 (bootstrap) : 5, 10, 15 documents          │
                 │     pré-annotés par Haiku → corrigés par un humain        │
                 │     → un modèle propose de meilleures descriptions        │
                 │       de champs (chap. 6)                                 │
                 │                                                          │
                 │  4. GEPA optimise le prompt sur les documents relus       │
                 │     (chap. 7), puis on le note sur les documents de test  │
                 │                                                          │
                 │  5. Apprentissage actif : on choisit les prochains        │
                 │     documents à annoter, ceux où le modèle hésite         │
                 │     (chap. 8)                                             │
                 │                                                          │
                 │  6. Suivi : rounds.csv, détection de plateau ──► stop ?   │
                 │        │ non                                              │
                 └────────┴──► retour à 4 avec plus de vérité terrain ──────┘
```

## Les modèles et leurs rôles

Tous les appels passent par **OpenRouter**, une passerelle qui expose de nombreux modèles
avec la même API (compatible OpenAI). Le projet utilise deux modèles pour quatre rôles :

| Rôle | Modèle | Ce qu'il fait |
|---|---|---|
| `extractor` | Qwen3.8-27B | l'extraction elle-même ; c'est **son** prompt qu'on optimise |
| `preannotator` | Claude Haiku 5.5 | propose des valeurs aux annotateurs humains |
| `diagnostic` | Claude Haiku 5.5 | regarde le document pour expliquer une erreur, en citant la zone |
| `reflection` | Claude Haiku 5.5 | réécrit les morceaux du prompt à partir des erreurs |

Pourquoi deux modèles différents ? Le modèle qui pré-annote ne doit pas être celui qu'on
évalue, sinon les annotateurs, influencés par ses propositions, confirmeraient ses propres
erreurs. Et Haiku sait regarder une image et raisonner, ce qui sert au diagnostic.

## Les splits, en bref

C'est le point le plus important à comprendre (détails au chapitre 3) :

- **`gepa_train`** : les documents sur lesquels GEPA apprend ;
- **`gepa_pareto`** : les documents sur lesquels GEPA *choisit* entre plusieurs versions du
  prompt. Comme il choisit grâce à eux, ils ne sont **pas** un vrai jeu de test ;
- **`heldout_blind`** : le vrai jeu de test, uniquement des documents réels, annotés sans
  voir les propositions du modèle, et jamais montrés à GEPA ni à aucun modèle qui écrit des
  prompts. Sur le Mac, faute de documents réels, un jeu de remplacement synthétique joue ce
  rôle : `heldout_standin`.

## Glossaire

| Terme | Signification ici |
|---|---|
| VLM | *Vision-Language Model* : un modèle qui lit des images et produit du texte |
| prompt | les instructions données au modèle avant le document |
| candidat | une version du prompt, sous forme de dictionnaire `{nom: texte}` (chap. 4) |
| composant | une clé de ce dictionnaire, par exemple `field:immatriculation` |
| gold / vérité terrain | la bonne réponse, validée par un humain |
| pré-annotation | une proposition du modèle, montrée à l'annotateur pour gagner du temps |
| round | un lot d'annotation (round 0 = test à l'aveugle, 1-3 = bootstrap, 4+ = actif) |
| metric call | une évaluation d'un document par une version du prompt (unité de budget GEPA) |
| held-out | « mis de côté » : jamais utilisé pour apprendre ni pour choisir |
| run | une exécution d'une commande, qui laisse un dossier dans `runs/` |

## Ce qui est hors périmètre

Pas de fine-tuning du modèle, pas de mode « raisonnement » pour l'extracteur, pas
d'interface d'annotation maison (on utilise Kili ou des fichiers JSON), pas de DSPy.

Chapitre suivant : [Installation et configuration](02-installation-et-configuration.md).
