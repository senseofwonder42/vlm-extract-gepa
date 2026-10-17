# Documentation en français

Cette documentation explique en détail le fonctionnement de `vlm-extract-gepa`. Elle
s'adresse à un data scientist qui arrive sur le projet : on suppose que vous connaissez
Python, pandas et les bases du machine learning (jeu d'entraînement, de validation, de
test), mais pas forcément les LLM, l'annotation de données ou GEPA.

## Ordre de lecture conseillé

| # | Chapitre | Vous y apprendrez |
|---|---|---|
| 1 | [Vue d'ensemble](01-vue-d-ensemble.md) | le problème, la boucle complète, le vocabulaire |
| 2 | [Installation et configuration](02-installation-et-configuration.md) | `uv`, `.env`, le YAML, les modèles et leurs rôles, les dossiers de run |
| 3 | [Les données](03-donnees.md) | spécimens réels, documents synthétiques, registre, découpage en splits |
| 4 | [Extraction et prompt](04-extraction-et-prompt.md) | le schéma, le prompt vu comme un « candidat », le client OpenRouter |
| 5 | [La métrique](05-metrique.md) | normalisation, comparaison champ par champ, types d'erreur, score |
| 6 | [Annotation](06-annotation.md) | le store, la vérité terrain, les backends local et Kili, le bootstrap |
| 7 | [GEPA](07-gepa.md) | comment GEPA améliore le prompt, notre adaptateur, les coûts |
| 8 | [Apprentissage actif et suivi](08-apprentissage-actif-et-suivi.md) | `vex select`, `vex report`, quand s'arrêter |
| 9 | [Guide pratique](09-guide-pratique.md) | un cycle complet pas à pas, dépannage, pièges fréquents |

Si vous n'avez qu'une heure : lisez les chapitres 1, 5 et 9, puis exécutez le notebook
[`notebooks/gepa_walkthrough.ipynb`](../../notebooks/gepa_walkthrough.ipynb).

## Autres documents

- [`docs/PLAN.md`](../PLAN.md) : le plan d'origine et toutes les décisions prises (en anglais).
- [`docs/gepa_guide.md`](../gepa_guide.md) : le guide GEPA en anglais, dont le chapitre 7 reprend l'essentiel.
- [`docs/kili_uncertainties.md`](../kili_uncertainties.md) : ce qui reste à vérifier sur Kili au bureau.

## Conventions

- Les commandes sont à lancer depuis la racine du dépôt.
- `vex` est la ligne de commande du projet : on l'appelle toujours via `uv run vex ...`.
- Les noms de code (fonctions, clés YAML, fichiers) restent en anglais, `comme ceci`.
- Les chiffres de coût et de performance cités viennent d'exécutions réelles faites le
  17 octobre 2026 ; ils changeront avec les prix et les modèles.
