# Projet DL sous TensorFlow

Ce dépôt vise une soumission propre et défendable pour un projet de deep learning orienté aide à la décision en contexte médical.

## Objectif métier

Le projet traite plusieurs types de données pour illustrer une vraie approche multimodale:
- données tabulaires pour le risque cardiovasculaire,
- texte clinique pour l'analyse de comptes rendus,
- images médicales pour la détection de pneumonie,
- une brique multimodale pour fusionner les sorties des modèles.

## Ce que le sujet demande, et où c'est couvert

| Exigence du sujet | Couverture dans le projet |
| --- | --- |
| Plusieurs types de données | `notebooks/Projet_Tabulaire_Cardiovasculaire.ipynb`, `notebooks/Detection_Textuelle.ipynb`, `notebooks/Detection_Pneumonie.ipynb`, `notebooks/text_audio.ipynb` |
| Modèles faits maison | MLP tabulaire, architectures textuelles et image construites dans les notebooks |
| Modèles preentraînés | ClinicalBERT, YAMNet et composants TensorFlow Hub dans les notebooks correspondants |
| Justification des choix | Chaque notebook documente les choix de pretraitement, d'architecture et de métriques |
| Etude comparative | `resultats_comparatifs.csv`, `synthese_tabulaire.json`, figures de comparaison dans `outputs/` |
| Outils TensorFlow | `tf.keras`, TensorFlow Hub, callbacks Keras, `SavedModel`/`.keras`, `tf.io`, `tf.image` |
| Bonus backend | `backend/` expose une API FastAPI pour la demo et les tests |

## Composition recommandee pour une soumission propre

Conserver:
- `backend/` pour l'API de demo et les tests,
- `notebooks/` pour les notebooks propres et executes,
- `outputs/` pour les figures et notebooks executes utiles a la soutenance,
- `data/sample/` pour les petits echantillons partageables,
- `README_notebooks.md` pour le guide d'execution et de demo,
- `requirements-notebooks.txt`, `run_notebooks.sh`, `run_notebooks.ps1` pour la reproductibilite,
- les fichiers de synthese et de comparaison: `resultats_comparatifs.csv`, `synthese_tabulaire.json`.

Exclure du commit:
- les environnements locaux (`.venv/`, `backend/.venv/`),
- les caches (`__pycache__/`, `.pytest_cache/`, `.ipynb_checkpoints/`),
- les scripts temporaires de diagnostic,
- les fichiers locaux sensibles (`.env`).

## Backend de demonstration

L'API est dans `backend/` et propose:
- `GET /api/v1/health`,
- `GET /api/v1/ready`,
- `POST /api/v1/predict/tabular`,
- `POST /api/v1/predict/text`,
- `POST /api/v1/predict/image`,
- `POST /api/v1/predict/multimodal`.

### Lancement local

```powershell
Set-Location backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

L'interface Swagger est disponible sur `http://127.0.0.1:8000/docs`.

## Validation du projet

- Les notebooks ont ete executes et les sorties utiles ont ete centralisees dans `outputs/`.
- Le backend charge les modeles TensorFlow disponibles et les tests passent.
- La fusion multimodale utilise une concaténation des scores de modalites avant prediction finale.
