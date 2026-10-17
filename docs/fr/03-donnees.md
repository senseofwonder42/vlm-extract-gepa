# 3. Les données

Trois sources de documents, un registre qui leur donne un identifiant stable, et un
découpage en splits figé une fois pour toutes.

## Les spécimens réels

Idéalement, on aurait 30 à 60 vrais justificatifs publics. En pratique, la recherche a
montré qu'il en existe très peu : les assureurs génèrent le Mémo à la demande dans l'espace
client, et les exemples publics sont surtout des images dans des articles.

La liste des sources candidates est dans `data/specimen_sources.yaml`. Chaque entrée a :

- un `status` : `proposed`, `approved` ou `excluded`. **Seules les entrées approuvées par
  un humain sont téléchargées** ;
- une évaluation `pii` (données personnelles) : `none`, `redacted` (floutées par
  l'éditeur) ou `suspected`. Une entrée `suspected` est refusée même si elle est approuvée ;
- une note de licence et une justification.

```bash
uv run vex data fetch-specimens
```

La commande vérifie le `robots.txt` de chaque site, attend 2 secondes entre deux
téléchargements, et inscrit chaque fichier dans `data/manifest.jsonl` (URL, empreinte
SHA-256, date, licence). Les fichiers eux-mêmes vont dans `data/raw/real/`, qui n'est pas
versionné : le manifeste suffit à reconstruire le jeu.

État actuel : **un seul** spécimen approuvé, un Mémo d'exemple publié par Leocare (nom
fictif, plaque `AB-123-CD`). Une carte allemande de Wikimedia a été exclue car elle montrait
le nom réel de son propriétaire.

## Les documents synthétiques

Pour avoir du volume, on génère 200 justificatifs fictifs :

```bash
uv run vex data synth
```

### Cinq mises en page

| Layout | Ressemble à |
|---|---|
| `memo` | un Mémo Véhicule Assuré (style carte d'application) |
| `carte_internationale` | la carte internationale blanche, cases numérotées 1 à 10 |
| `carte_verte` | l'ancienne carte verte avec son « papillon » à coller sur le pare-brise |
| `attestation_lettre` | une attestation rédigée comme une lettre |
| `attestation_tableau` | un formulaire avec tableau des garanties à cocher |

Les assureurs sont **fictifs** (« Mutuelle Imaginaire de l'Ouest », « Assur'Mirage »…) et
un test vérifie qu'aucun nom ne contient celui d'un assureur réel. Les logos sont des formes
géométriques dessinées par le code. Les personnes et adresses viennent de Faker (`fr_FR`).

### Ce qui rend les documents réalistes

- **Des pièges** : numéro de sociétaire (à ne pas confondre avec le numéro de contrat),
  numéro de série du véhicule (VIN), date d'émission du document, mentions légales,
  nom d'un courtier ;
- **des variantes de format** : dates « 01/02/2025 », « 1er février 2025 », « 01.02.2025 » ;
  plaques avec ou sans tirets ; anciennes plaques FNI ;
- **des cas limites** : 10 documents à deux véhicules (on ne garde que le premier), 15 de
  deux pages, des champs absents (le Mémo n'a pas toujours d'adresse) ;
- **des scans dégradés** (65 % des documents) : rotation de quelques degrés et parfois de
  90°/180°/270°, flou, contraste, bruit, compression JPEG.

Rendu technique : modèle Jinja2 en HTML → PDF avec `pymupdf.Story` → image → dégradation.
On a choisi PyMuPDF plutôt que WeasyPrint ou Playwright parce qu'il était déjà installé, ne
demande aucune bibliothèque système ni navigateur, et donne exactement les mêmes octets à
chaque exécution.

### `synthetic_meta` : les valeurs du générateur

Le générateur sait ce qu'il a écrit. Il l'enregistre dans
`data/synthetic_meta/meta.jsonl`, par exemple `immatriculation: "LC-280-DM"` même si le
document affiche « LC 280 DM ».

> **Règle absolue : ces valeurs ne servent jamais d'étiquettes**, ni pour entraîner ni pour
> évaluer.

Pourquoi ? Parce qu'elles décrivent l'*intention* du générateur, pas ce qu'un humain lit
réellement sur l'image (un scan peut rendre un champ illisible). Elles ne servent qu'aux
tests et à `vex data sanity`, un contrôle rapide qui note un prompt sur quelques documents
synthétiques. Un test vérifie qu'aucun autre module ne les lit.

## Le registre : un identifiant stable par document

```bash
uv run vex data register
```

Chaque fichier reçoit un `doc_id` = **l'empreinte SHA-256 de son contenu**. Avantages :
renommer un fichier ne change rien, et un doublon téléchargé deux fois n'est enregistré
qu'une fois. Le registre (`data/registry.jsonl`) note aussi l'origine (`real` ou
`synthetic`), le nombre de pages et, pour les synthétiques, le layout (lu dans le nom du
fichier).

### Une leçon apprise : le déterminisme

Puisque l'identifiant dépend des octets, le générateur doit produire **exactement** les
mêmes octets à chaque exécution. Ce n'était pas le cas au début : l'encodeur JPEG de Pillow
produisait parfois des fichiers différents à partir des mêmes pixels (environ une
exécution sur deux, sur Mac Apple Silicon). Les `doc_id` des scans auraient changé à chaque
régénération, et le découpage en splits serait devenu faux. L'encodage JPEG passe donc par
PyMuPDF, et un test lance le générateur dans plusieurs processus séparés pour vérifier que
les octets sont identiques.

À retenir : quand un identifiant dépend d'un hash de fichier, testez la reproductibilité
**entre processus**, pas seulement dans un même processus.

## Les splits

```bash
uv run vex data split
```

Le résultat est figé dans `data/splits.yaml`, versionné. Il n'est jamais modifié à la main.

### Pourquoi trois ensembles et non deux

En machine learning classique, on a *train / validation / test*. Ici, GEPA utilise son
ensemble de validation pour **choisir** la meilleure version du prompt. Tout ce qui sert à
choisir devient optimiste : ce n'est plus un vrai test. D'où :

| Split | Rôle | Vu par GEPA ? |
|---|---|---|
| `gepa_train` | apprendre : GEPA y voit les erreurs et teste ses corrections | oui |
| `gepa_pareto` | choisir entre les versions du prompt (la « validation » de GEPA) | oui |
| `heldout_blind` | **le vrai test** : documents réels, annotés à l'aveugle | **jamais** |

### Comment les documents sont répartis

- `heldout_blind` reçoit la moitié des documents **réels**, au plus 15 (`splits.heldout_blind_size`).
  Sous 8 documents, un avertissement signale que le score de test sera peu fiable ;
- les autres documents vont dans le « seau » `gepa_train` ou `gepa_pareto` (40 % pour ce
  dernier, `splits.pareto_fraction`).

La position d'un document ne dépend que de `sha256(seed, doc_id)`, pas de l'ordre dans
lequel les fichiers sont arrivés. Et **l'affectation est incrémentale** : si on ajoute des
documents plus tard, les anciens ne bougent jamais, seuls les nouveaux sont placés. C'est ce
qui permettra d'ajouter des documents de l'entreprise au bureau sans tout casser.

### Seau ≠ ensemble utilisé

Un document du seau `gepa_train` n'est utilisé par GEPA **que s'il a une vérité terrain**
(annotation relue). L'ensemble réellement utilisé est donc :

```
gepa_train effectif  = seau gepa_train  ∩ documents annotés
gepa_pareto effectif = seau gepa_pareto ∩ documents annotés
```

Les ensembles grandissent à chaque round d'annotation, alors que l'affectation reste figée.
Un document non relu ne peut jamais y entrer.

### Le cas du Mac : `heldout_standin`

Avec un seul document réel, `heldout_blind` est vide (la moitié de 1, arrondie, fait 0). On
a donc décidé d'un remplaçant temporaire :

- tout le layout `carte_verte` est **réservé** (`splits.standin_layouts`) ;
- 15 de ces documents forment `heldout_standin`, annoté à l'aveugle et traité exactement
  comme `heldout_blind` ;
- les 25 autres sont mis dans `reserved_unused` et ne servent jamais.

Ce score mesure la capacité du prompt à **généraliser à une mise en page jamais vue**, pas
sa performance sur de vrais documents. Au bureau, on mettra `standin_layouts: []` et on
ajoutera des documents réels anonymisés.

État actuel : `heldout_blind` = 0, `heldout_standin` = 15, `reserved_unused` = 25,
seau `gepa_train` = 96, seau `gepa_pareto` = 65.

Chapitre suivant : [Extraction et prompt](04-extraction-et-prompt.md).
