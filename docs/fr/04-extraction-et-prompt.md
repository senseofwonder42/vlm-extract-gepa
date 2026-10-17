# 4. Extraction et prompt

Ce chapitre suit le trajet d'un document : du fichier PDF jusqu'au JSON validé.

## Le schéma : `AttestationAuto`

Le contrat de sortie est un modèle Pydantic dans `src/vex/schema.py`. Chaque champ porte une
**description** très précise, en français, par exemple pour l'immatriculation :

> Numéro d'immatriculation du véhicule assuré (« Immatriculation », case 5 de la carte
> internationale). Format SIV (depuis 2009) : AA-123-AA ; ancien format FNI : 1234 AB 56 ou
> 123 ABC 75. Recopier en majuscules, avec les tirets pour le SIV. Ne pas confondre avec le
> numéro VIN (17 caractères) ni le numéro de contrat. Si plusieurs véhicules sont listés,
> prendre uniquement le PREMIER. […] Si le champ est absent ou illisible → null.

Ces descriptions disent où chercher, quoi exclure, quel format rendre et quand répondre
`null`. Elles sont en français parce qu'elles servent aussi de consignes aux annotateurs
dans Kili : humains et modèle lisent les mêmes règles.

Le schéma valide aussi la sortie :

- une chaîne vide ou une liste vide devient `null` ;
- les dates doivent être au format ISO `AAAA-MM-JJ`. Une date française comme
  « 01/02/2025 » est **refusée**, ce qui déclenche une tentative de réparation (voir plus bas) ;
- un champ inconnu est refusé.

**Plusieurs véhicules sur un document** : on n'extrait que le premier. La règle est écrite
dans les descriptions de `immatriculation` et `marque_vehicule`, pour que le modèle et les
annotateurs l'appliquent de la même façon.

## Le prompt vu comme un « candidat »

C'est l'idée centrale du projet. Le prompt n'est pas un long texte monolithique, c'est un
**dictionnaire** :

```python
{
    "instructions":          "Tu extrais des champs d'un justificatif d'assurance ...",
    "field:assureur":        "Nom de la compagnie d'assurance qui porte le risque ...",
    "field:numero_contrat":  "Numéro du contrat, de police ou d'adhésion ...",
    ...                      # un composant par champ, 11 au total
    "field:garanties":       "Liste des garanties souscrites ...",
}
```

Ce dictionnaire s'appelle un **candidat** (vocabulaire de GEPA) et chaque clé un
**composant**. Pourquoi découper ainsi ?

- GEPA peut corriger la règle d'un seul champ sans toucher aux autres ;
- chaque changement se lit comme un diff court et compréhensible ;
- on peut savoir quel champ s'est amélioré grâce à quelle réécriture.

Le candidat de départ (« seed ») est construit à partir du schéma : les instructions
générales plus la description de chaque champ.

```bash
uv run vex prompt seed     # écrit prompts/seed.437afbe1.json
```

Les candidats sont enregistrés dans `prompts/<nom>.<hash8>.json`, où le hash ne dépend que
du contenu. Si quelqu'un modifie le fichier à la main, le hash ne correspond plus et le
chargement échoue : on sait toujours exactement quel prompt a produit quel résultat.

### Du candidat au message envoyé

`vex.prompt.render(candidate)` produit deux choses :

1. un **prompt système** : les instructions, puis la liste « Champs à extraire » avec la
   description de chaque champ ;
2. un **schéma JSON strict** dont les descriptions de champs viennent du candidat (et non du
   code Python).

Le message utilisateur contient ensuite les pages du document sous forme d'images. Un PDF
est converti en PNG avec PyMuPDF à 150 DPI (`render.dpi`), et on envoie au plus 3 pages
(`render.max_pages`). Une image PNG ou JPEG est envoyée telle quelle, sans réencodage qui
ajouterait des artefacts.

## La sortie structurée

On veut un JSON valide à coup sûr. Deux chemins existent (`structured_output` dans la
configuration de chaque rôle) :

- **`json_schema`** : on envoie le schéma dans le champ `response_format` de la requête. Le
  fournisseur contraint alors la génération pour qu'elle respecte le schéma. C'est le chemin
  le plus fiable ;
- **`prompt`** : si le fournisseur ne sait pas faire, on ajoute le schéma à la fin du
  prompt et on espère que le modèle le respecte, puis on extrait le JSON de la réponse (en
  tolérant les balises ```` ```json ````).

Avec `auto` (par défaut), le client interroge une fois la liste des fournisseurs du modèle
sur OpenRouter et choisit `json_schema` si le fournisseur épinglé annonce le support des
sorties structurées. Le chemin choisi est écrit dans le log et dans le résultat.

### Une seule tentative de réparation

Si la réponse ne se lit pas comme du JSON ou ne passe pas la validation (par exemple une
date « 01/02/2025 »), le client renvoie au modèle sa réponse et le message d'erreur, en
demandant le JSON corrigé. **Une seule fois.** Si la deuxième réponse échoue aussi :

- l'extraction revient avec `values = None` et le message d'erreur ;
- ce n'est pas une exception : la chaîne de traitement continue ;
- la métrique comptera tous les champs comme des oublis et le document aura 0 (chapitre 5).

En pratique, sur les essais réels, ni Qwen ni Haiku n'ont eu besoin de réparation sur le
chemin `json_schema`.

## Essayer une extraction

```bash
uv run vex extract data/raw/synthetic/syn_0010_memo.jpg
```

Par défaut, la commande utilise le rôle `extractor_eval` (température 0) et le candidat de
départ. Options utiles :

```bash
uv run vex extract doc.pdf --candidate prompts/bootstrap-r1.ab12cd34.json
uv run vex extract doc.pdf --role preannotator          # Haiku au lieu de Qwen
```

Le résultat (`ExtractionResult`, dans `src/vex/extract.py`) contient : les valeurs, les
réponses brutes de chaque tentative, le chemin de sortie structurée, un indicateur de
réparation, l'erreur éventuelle, le hash du candidat utilisé et le coût. Le tout est écrit
dans `runs/<...>-extract-<hash>/result.json`.

## Où regarder dans le code

| Fichier | Contenu |
|---|---|
| `src/vex/schema.py` | `AttestationAuto`, les descriptions, le schéma JSON strict |
| `src/vex/prompt.py` | candidat de départ, rendu, sauvegarde et chargement avec hash |
| `src/vex/llm/media.py` | PDF → images |
| `src/vex/llm/client.py` | client OpenRouter : sortie structurée, réparation, cache, réessais, budget |
| `src/vex/extract.py` | `extract_document` : un document + un candidat → `ExtractionResult` |

Chapitre suivant : [La métrique](05-metrique.md).
