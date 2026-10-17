# 2. Installation et configuration

## Installer

Le projet utilise [`uv`](https://docs.astral.sh/uv/) pour gérer Python 3.12 et les
dépendances. On n'installe rien « à la main » avec `pip`, et on n'édite jamais les
dépendances dans `pyproject.toml` : on passe par `uv add <paquet>`.

```bash
uv sync                    # crée .venv avec toutes les dépendances
uv sync --extra embed      # optionnel : embeddings d'images pour `vex select` (chap. 8)
cp .env.example .env       # puis renseigner la clé OpenRouter
uv run vex doctor          # vérifie la clé et les fournisseurs
```

## Les secrets : `.env`

Les clés d'API ne sont **que** dans `.env`, qui est ignoré par git :

```
OPENROUTER_API_KEY=sk-or-...
KILI_API_KEY=...           # au bureau seulement
KILI_API_ENDPOINT=...      # idem
```

Elles sont lues par `vex.config.Settings` (pydantic-settings), ne sont jamais écrites dans
un dossier de run, n'entrent pas dans le hash de configuration, et un filtre de log masque
tout ce qui ressemble à une clé (`vex/logs.py`).

## La configuration : `configs/default.yaml`

Tout ce qui se règle est dans ce fichier : modèles, paramètres, chemins, splits, poids des
champs, budget, GEPA, apprentissage actif. Il est validé par des modèles Pydantic
(`src/vex/config.py`) qui **refusent les clés inconnues** : une faute de frappe fait
échouer la commande au lieu d'être silencieusement ignorée.

### Surcharger une clé depuis la ligne de commande

N'importe quelle clé peut être modifiée pour une exécution, avec `--set` placé **avant**
la commande :

```bash
uv run vex --set models.extractor.params.temperature=0.2 extract doc.pdf --role extractor
uv run vex --set budget.max_usd=2 --set gepa.use_merge=false optimize
uv run vex --set metric.fuzzy_fields=[assureur] config show
```

La valeur est lue comme du YAML : `0.2` donne un nombre, `true` un booléen, `[a, b]` une
liste. `uv run vex config show` affiche la configuration finale et son hash.

## Les rôles de modèles

Chaque rôle a la même structure :

```yaml
models:
  extractor:
    model: qwen/qwen3.8-27b
    structured_output: auto
    params: {temperature: 0.7, top_p: 0.8, top_k: 20, presence_penalty: 1.5,
             repetition_penalty: 1.0, max_tokens: 1500, seed: 0}
    extra_body:
      reasoning: {effort: none}
      provider: {order: [parasail], allow_fallbacks: false, require_parameters: true}
```

`params` et `extra_body` sont envoyés **tels quels** à OpenRouter. On peut donc ajouter
n'importe quel paramètre accepté par OpenRouter sans toucher au code.

### Pourquoi ces valeurs ?

- **`reasoning: {effort: none}`** pour l'extracteur. Sur OpenRouter, Qwen3.8-27B
  « réfléchit » par défaut (effort `xhigh`) : il écrit un long raisonnement avant de
  répondre, et on paie ces tokens. On l'a vérifié : sur une question triviale, 149 tokens
  de raisonnement par défaut, 0 avec `effort: none`.
- **Les paramètres d'échantillonnage** sont ceux recommandés par la fiche officielle du
  modèle pour le mode sans raisonnement. Un paramètre manque : `min_p`. La fiche recommande
  `min_p: 0.0`, qui est la valeur neutre, et le fournisseur Parasail ne l'accepte pas
  (OpenRouter renvoyait une erreur 404). On l'a donc retiré.
- **`provider.order: [parasail]`** : vingt fournisseurs servent ce modèle, avec des
  quantifications différentes (de fp4 à bf16) qui donnent des réponses différentes. On en
  fixe un pour que les scores soient comparables d'une exécution à l'autre.
  `require_parameters: true` interdit à OpenRouter de router vers un fournisseur qui
  ignorerait un de nos paramètres (par exemple le format JSON imposé).
- **`extractor_eval`** est l'extracteur avec `temperature: 0` et `presence_penalty: 0`.
  C'est lui qu'on utilise pour **noter** un prompt : à température 0, deux exécutions
  donnent le même résultat, donc une différence de score vient du prompt et non du hasard.
  Le rôle `extractor` (paramètres officiels) sert uniquement à l'échantillonnage du
  chapitre 8.
- **Les trois rôles Haiku** gardent un raisonnement d'effort `medium`. Il est adaptatif : on
  a mesuré entre 0 et 306 tokens de raisonnement selon la difficulté, et 0 sur une simple
  extraction. Haiku n'accepte pas de `temperature`.

`uv run vex doctor` vérifie que chaque fournisseur épinglé accepte bien les paramètres
envoyés. C'est cette commande qui a révélé le problème de `min_p`.

## Les dossiers de run

Chaque commande qui appelle un modèle crée un dossier :

```
runs/20261017T191630262707Z-optimize-aef58709/
├── config.resolved.yaml   # la configuration exacte utilisée (après les --set)
├── config.sha256          # son hash
├── log.txt                # le log détaillé (niveau DEBUG)
├── cost.json              # tokens et dollars, par rôle et par appel
└── ...                    # les sorties propres à la commande
```

Le nom contient l'heure UTC, la commande et les 8 premiers caractères du hash de
configuration. Deux runs avec la même configuration ont le même suffixe. `runs/` n'est
pas versionné.

## Le cache disque

Chaque réponse de modèle est enregistrée dans `.cache/llm/`. La clé de cache est le hash
SHA-256 de tout ce qui détermine la réponse : modèle, paramètres, `extra_body`, format de
sortie et messages (les images y sont remplacées par leur propre hash). Si on relance
exactement la même requête, la réponse vient du disque et coûte 0 $.

Conséquences pratiques :

- relancer un notebook ou une évaluation est presque gratuit ;
- changer un seul mot du prompt change la clé : le nouvel appel est payant ;
- `uv run vex --no-cache <commande>` force de vrais appels (par exemple pour mesurer une
  variabilité).

## Budget, coûts, réessais

- **Budget** : `budget.max_usd` (5 $ par défaut) est vérifié **avant** chaque appel. Une fois
  atteint, l'appel est refusé (`BudgetExceeded`). Le coût exact vient du champ
  `usage.cost` qu'OpenRouter renvoie avec chaque réponse.
- **Concurrence** : au plus `client.concurrency` (8) appels en parallèle.
- **Réessais** : sur une erreur 429 (trop de requêtes), 408 ou 5xx, le client réessaie
  jusqu'à 5 fois avec une attente exponentielle (1 s, 2 s, 4 s… plafonnée à 60 s), en
  respectant l'en-tête `Retry-After` s'il est fourni. Une erreur 400 (requête invalide)
  n'est pas réessayée.

Ordres de grandeur mesurés :

| Appel | Coût |
|---|---|
| extraction Qwen, 1 page | ~0,0012-0,0016 $ (~0,0006 $ si le fournisseur réutilise son propre cache) |
| extraction Haiku, 1 page | ~0,0009 $ |
| diagnostic Haiku | ~0,0006 $ |
| réécriture par Haiku | ~0,001-0,002 $ |

Chapitre suivant : [Les données](03-donnees.md).
