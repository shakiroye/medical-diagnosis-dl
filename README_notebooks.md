# Notebooks - Reproducibility & Execution

Ce dossier contient les instructions pour exécuter et reproduire les notebooks du projet.

Principes:
- Les notebooks sont la source unique d'exploration et de résultats pour chaque modalité.
- Ne pas commiter de jeux de données volumineux; stocker des petits échantillons dans `data/sample/` et fournir des liens et scripts pour récupérer les jeux complets.

Exécution reproducible (pré-requis):
- Python 3.11 (ou proche)
- Installer les dépendances listées dans `requirements-notebooks.txt`

Exécution (rapide):

```bash
python -m pip install -r requirements-notebooks.txt
# exécute la liste de notebooks définie dans `notebooks_to_run.txt`
./run_notebooks.sh
```

Ou sous PowerShell:

```powershell
python -m pip install -r requirements-notebooks.txt
.\run_notebooks.ps1
```

Résultats et outputs:
- Les sorties (figures, notebooks exécutés, exports) seront placés dans `outputs/`.
- Pour réexécuter un notebook spécifique sans papermill, ouvrir le notebook dans JupyterLab / Jupyter Notebook et exécuter "Run all cells".

Data policy:
- Les jeux complets (Kaggle, UCR) **ne doivent pas** être committés. Fournir un script ou instructions pour téléchargement à partir des sources officielles (voir `data/sample/README.md`).

## Mini guide de démo backend

Ce guide sert à vérifier rapidement que l'API backend est bien fonctionnelle pendant la présentation.

Pré-requis:
- Le dossier `backend/` contient les modèles dans `backend/models/`.
- Python et les dépendances backend sont installés dans la venv du projet.

Lancement local:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Tests rapides:
- `GET /api/v1/health` doit renvoyer `{"status":"ok"}`.
- `GET /api/v1/ready` doit indiquer que `tabular`, `text`, `image` et `multimodal` sont disponibles et chargés.
- `POST /api/v1/predict/tabular`
- `POST /api/v1/predict/text`
- `POST /api/v1/predict/image`
- `POST /api/v1/predict/multimodal`

Exemples de payloads:

```json
{
	"patient_id": "P001",
	"features": {"age": 65, "sex": 1, "cp": 3, "chol": 240, "thalach": 150}
}
```

```json
{
	"patient_id": "P002",
	"text": "Patient reports cough and chest pain with mild dyspnea.",
	"language": "en"
}
```

```json
{
	"patient_id": "P003",
	"image_base64": "<base64_png>",
	"image_type": "chest_xray"
}
```

```json
{
	"patient_id": "P004",
	"tabular": {"patient_id": "P004", "features": {"age": 58, "sex": 0, "cp": 2, "chol": 210, "thalach": 140}},
	"text": {"patient_id": "P004", "text": "Short report with fever and cough.", "language": "en"},
	"image": {"patient_id": "P004", "image_base64": "<base64_png>", "image_type": "chest_xray"}
}
```

Pour une démo propre:
- Ouvrir `http://127.0.0.1:8000/docs` et exécuter les endpoints depuis Swagger.
- Montrer d'abord `/health`, puis `/ready`, puis un endpoint de prédiction par modalité.
- Mentionner que les réponses incluent `modality`, `label`, `confidence`, `model_name` et `metadata`.

## Avant de push

Pour un dépôt propre, ne garder que les fichiers réellement voulus:
- Conserver le code source, la documentation et les artefacts utiles à la démo.
- Éviter de committer les environnements locaux, caches, fichiers temporaires et sorties intermédiaires inutiles.
- Vérifier `git status` avant le push et retirer du stage tout fichier généré qui ne doit pas versionner.
- Si un fichier généré est indispensable à la présentation, le documenter clairement dans le README.
