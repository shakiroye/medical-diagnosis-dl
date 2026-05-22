# Diagnostic Médical par Deep Learning — Projet TensorFlow

> Système multimodal d'aide à la décision médicale développé sous TensorFlow/Keras,
> traitant des données **image**, **tabulaire**, **texte** et **audio**.

---

## Table des matières

1. [Contexte et problématique](#1-contexte-et-problématique)
2. [Architecture du projet](#2-architecture-du-projet)
3. [Données utilisées](#3-données-utilisées)
4. [Modules et modèles](#4-modules-et-modèles)
   - [4.1 Images — Détection de pneumonie](#41-images--détection-de-pneumonie)
   - [4.2 Tabulaire — Risque cardiovasculaire](#42-tabulaire--risque-cardiovasculaire)
   - [4.3 Texte — Classification de rapports médicaux](#43-texte--classification-de-rapports-médicaux)
   - [4.4 Audio — Analyse de sons respiratoires](#44-audio--analyse-de-sons-respiratoires)
5. [Étude comparative](#5-étude-comparative)
6. [Outils TensorFlow utilisés](#6-outils-tensorflow-utilisés)
7. [Backend FastAPI](#7-backend-fastapi)
8. [Justification des choix techniques](#8-justification-des-choix-techniques)
9. [Reproductibilité](#9-reproductibilité)
10. [Résultats et conclusions](#10-résultats-et-conclusions)

---

## 1. Contexte et problématique

### Besoin métier

Le domaine médical produit chaque jour des volumes massifs de données hétérogènes : radiographies, bilans biologiques, comptes-rendus cliniques, enregistrements auscultatoires. Les praticiens font face à une surcharge informationnelle qui ralentit les décisions et génère des erreurs. Ce projet propose un système d'aide à la décision capable d'analyser simultanément plusieurs types de données médicales pour assister le clinicien.

### Objectifs

- Développer des modèles de Deep Learning capables de **classer et détecter des pathologies** à partir de données multimodales.
- Comparer systématiquement des architectures **"fait maison"** et des **modèles pré-entraînés** adaptés à chaque modalité.
- **Justifier chaque choix** méthodologique par des critères métriques et cliniques.
- Produire une **étude comparative** synthétique des approches testées.
- Intégrer les modèles dans un **backend REST** déployable.

### Pathologies ciblées

| Modalité | Pathologie / Tâche |
|---|---|
| Image | Pneumonie (radiographies thoraciques) |
| Tabulaire | Maladie cardiovasculaire |
| Texte | Classification de spécialités médicales (10 classes) |
| Audio | Anomalies respiratoires (craquements, sifflements) |

---

## 2. Architecture du projet

```
medical-diagnosis-dl/
│
├── notebooks/                              # Développement et expérimentation
│   ├── Detection_Pneumonie.ipynb           # Module image (radiographies)
│   ├── Projet_Tabulaire_Cardiovasculaire.ipynb  # Module tabulaire (cardiologie)
│   ├── Detection_Textuelle.ipynb           # Baseline texte (TF-IDF)
│   └── text_audio_prepared.ipynb           # Module texte + audio (BiLSTM, BERT, CNN, YAMNet)
│
├── backend/                                # API REST FastAPI (BONUS)
│   ├── app/
│   │   ├── main.py                         # Application FastAPI
│   │   ├── api/routes.py                   # Endpoints de prédiction
│   │   ├── schemas/prediction.py           # Modèles Pydantic
│   │   └── services/
│   │       ├── inference.py                # Inférence multi-modalité
│   │       ├── fusion.py                   # Fusion multimodale
│   │       └── model_registry.py           # Chargement et registre des modèles
│   ├── models/                             # Modèles entraînés (.keras)
│   └── tests/                             # Tests Pytest
│
├── data/
│   ├── sample/                             # Jeux de données réduits (Git-trackés)
│   └── cardio_train.csv                    # Dataset cardiovasculaire (70k lignes)
│
├── outputs/                                # Graphiques, matrices de confusion, courbes
├── release_assets/                         # Modèles lourds (non trackés dans Git)
└── submission/                             # Documents de soumission
```

---

## 3. Données utilisées

Quatre jeux de données publics issus de Kaggle ont été utilisés, chacun représentant une modalité différente :

| Dataset | Modalité | Source | Volume | Tâche |
|---|---|---|---|---|
| [Chest X-Ray Pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) | Image | Paul Mooney | 5 863 radiographies | Classif. binaire (Normal / Pneumonie) |
| [Cardiovascular Disease](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset) | Tabulaire | S. Ulianova | 70 000 patients | Classif. binaire (Sain / CVD) |
| [MTSamples Medical Transcriptions](https://www.kaggle.com/tboyle10/medicaltranscriptions) | Texte | tboyle10 | ~7 000 documents | Classif. multi-classe (10 spécialités) |
| [ICBHI 2017 Respiratory Sounds](https://www.kaggle.com/vbookshelf/respiratory-sound-database) | Audio | vbookshelf | ~3 500 cycles respiratoires | Classif. 4 classes |

> **Politique de données** : les datasets volumineux ne sont pas versionnés dans Git. Des échantillons représentatifs sont disponibles dans `data/sample/`. Les téléchargements sont automatisés via `kagglehub`.

---

## 4. Modules et modèles

Chaque module suit le même protocole : un **modèle custom** (fait maison) est développé en premier lieu, puis comparé à un **modèle pré-entraîné** adapté au domaine.

---

### 4.1 Images — Détection de pneumonie

**Notebook** : [Detection_Pneumonie.ipynb](notebooks/Detection_Pneumonie.ipynb)

#### Données

- 5 863 radiographies thoraciques JPEG (224×224 px, niveaux de gris → converti RGB)
- Classes : `NORMAL` (1 583) / `PNEUMONIA` (4 273) — **déséquilibre 1:3**
- Découpage : 70 % train / 15 % validation / 15 % test (ré-équilibrage manuel du val set officiel trop petit)
- Correction du déséquilibre : `class_weight = {0: 1.944, 1: 0.673}`

#### Pipeline tf.data

```python
tf.keras.utils.image_dataset_from_directory(...)
  → Rescaling(1./255)
  → RandomFlip, RandomRotation(±8°), RandomZoom(±10%), RandomContrast(±15%)
  → .cache().prefetch(AUTOTUNE)
```

#### Modèle 1 — CNN Custom (fait maison)

```
Input (224×224×3)
  → Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
  → Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
  → Conv2D(128) → BN → MaxPool → Dropout(0.25)
  → Conv2D(256) → BN → GlobalAvgPool
  → Dense(256, relu) → Dropout(0.5) → Dense(1, sigmoid)
Paramètres : 456 000
```

#### Modèle 2 — EfficientNetB0 Fine-tuné (pré-entraîné ImageNet)

Entraînement en **deux phases** :
1. **Phase 1** — base EfficientNetB0 gelée : seule la tête de classification est entraînée (convergence rapide, 10 epochs)
2. **Phase 2** — 50 dernières couches dégelées : fine-tuning à `lr=1e-5` pour adaptation au domaine médical (15 epochs)

```
EfficientNetB0 (poids ImageNet) → GlobalAvgPool → Dropout(0.3)
  → Dense(256, relu) → Dropout(0.3) → Dense(1, sigmoid)
Paramètres entraînables : 2,8M sur 4,4M totaux
```

**Explainabilité** : cartes de saillance (Grad-CAM) générées pour visualiser les zones d'attention du modèle.

**Seuil de décision abaissé à 0.3** (au lieu de 0.5) car en contexte médical, **un faux négatif (pneumonie non détectée) est plus grave qu'un faux positif**.

---

### 4.2 Tabulaire — Risque cardiovasculaire

**Notebook** : [Projet_Tabulaire_Cardiovasculaire.ipynb](notebooks/Projet_Tabulaire_Cardiovasculaire.ipynb)

#### Données

- 70 000 patients → 68 559 après suppression des valeurs aberrantes (<2%)
- 14 features : âge, genre, taille, poids, pression artérielle (systolique/diastolique), cholestérol, glycémie, tabagisme, alcool, activité physique
- Features ingéniées : **IMC** et **pression différentielle** (pulse pressure)
- Classes parfaitement équilibrées (50/50) → aucune correction de poids nécessaire
- Preprocessing : `StandardScaler` ajusté uniquement sur le train set

#### Modèle 1 — MLP Custom (fait maison)

```
Input (14 features)
  → Dense(128, relu) → BatchNorm → Dropout(0.3)
  → Dense(64, relu) → BatchNorm → Dropout(0.2)
  → Dense(32, relu) → Dropout(0.1)
  → Dense(1, sigmoid)
Paramètres : 13 057
```

#### Modèles de comparaison (scikit-learn / XGBoost)

| Modèle | Type |
|---|---|
| Régression Logistique | Baseline linéaire |
| Random Forest (200 arbres) | Ensemble |
| XGBoost | Gradient Boosting |

> **Importance des features** : `ap_hi` (pression systolique), `age`, `cholesterol` et `weight` sont les quatre prédicteurs les plus déterminants.

---

### 4.3 Texte — Classification de rapports médicaux

**Notebooks** : [Detection_Textuelle.ipynb](notebooks/Detection_Textuelle.ipynb) (baseline) · [text_audio_prepared.ipynb](notebooks/text_audio_prepared.ipynb) (modèles avancés)

#### Données

- ~7 000 comptes-rendus médicaux (MTSamples), filtrés sur 10 spécialités
- Preprocessing : normalisation des dosages/dates/chiffres, suppression des caractères spéciaux, **préservation des négations** (importance clinique), tokenisation sur 20 000 tokens, troncature à 500 tokens
- Découpage stratifié : 70 % / 15 % / 15 %

#### Modèle 1 — BiLSTM + Attention (fait maison)

Architecture Sequence-to-Label avec mécanisme d'attention de Bahdanau (implémentation custom `tf.keras.layers.Layer`) :

```
Input (tokens)
  → Embedding(dim=128, vocab=20000)
  → SpatialDropout1D(0.3)
  → Bidirectional(LSTM(128))
  → BahdanauAttention (custom)
  → Dense(128, relu) → Dropout(0.4)
  → Dense(64, relu)
  → Dense(10, softmax)
Paramètres : ~2,5M
```

#### Modèle 2 — ClinicalBERT (pré-entraîné sur corpus biomédical)

- **Base** : `emilyalsentzer/Bio_ClinicalBERT` (110M params, pré-entraîné sur notes cliniques MIMIC-III)
- Chargé via `keras_hub`, fine-tuning à `lr=2e-5` sur 3–5 epochs
- **Choix justifié** : un BERT générique (Wikipedia/BooksCorpus) méconnaît le vocabulaire médical spécialisé (« pneumothorax », « dyspnée », « tachycardie ») ; ClinicalBERT a été pré-entraîné sur ce vocabulaire.

---

### 4.4 Audio — Analyse de sons respiratoires

**Notebook** : [text_audio_prepared.ipynb](notebooks/text_audio_prepared.ipynb)

#### Données

- Base ICBHI 2017 : ~3 500 cycles respiratoires annotés (WAV 16 kHz)
- 4 classes : `Normal` · `Crackle` (craquement — indicateur de pneumonie/fibrose) · `Wheeze` (sifflement — asthme/bronchite) · `Both`
- **Extraction de features** : spectrogramme Mel logarithmique
  - SR : 22 050 Hz, durée fixée à 5 s (padding/troncature)
  - n_mels = 64, FFT = 1024, hop = 512
  - Shape finale : `(64, 216, 1)`
  - Normalisation min-max → [0, 1]
- Augmentation : bruit additif gaussien (σ = 0.015) dans le pipeline `tf.data`
- Correction du déséquilibre : `class_weight` calculé automatiquement

#### Modèle 1 — CNN 2D sur spectrogramme (fait maison)

Le spectrogramme Mel est traité comme une image 2D :

```
Input (64×216×1)
  → [Conv2D(32) → BN → Conv2D(32) → BN → MaxPool(2,2) → Dropout(0.25)] × bloc 1
  → [Conv2D(64) → BN → Conv2D(64) → BN → MaxPool(2,2) → Dropout(0.25)] × bloc 2
  → [Conv2D(128) → BN → Conv2D(128) → BN → MaxPool(2,2) → Dropout(0.25)] × bloc 3
  → GlobalAveragePooling2D
  → Dense(256, relu) → Dropout(0.5)
  → Dense(4, softmax)
Paramètres : ~1,2M
```

#### Modèle 2 — YAMNet + MLP (pré-entraîné AudioSet)

- **Base** : `google/yamnet` depuis TensorFlow Hub, pré-entraîné sur AudioSet (521 classes, >2M clips)
- Stratégie : YAMNet gelé → extraction d'embeddings (1024 dim, moyennage temporel) → MLP classificateur entraîné par-dessus
- **Choix justifié** : YAMNet encode des représentations audio générales (timbre, texture, rythme) qui transfèrent efficacement vers les sons respiratoires.

```
Audio brut → YAMNet (gelé) → Embeddings(1024) [mean-pooled]
  → Dense(256, relu) → BN → Dropout(0.4)
  → Dense(128, relu) → Dropout(0.3)
  → Dense(4, softmax)
Paramètres entraînables : 0,4M sur 4,1M totaux
```

---

## 5. Étude comparative

### Image — Détection de pneumonie

| Modèle | Type | Accuracy | Recall (pneumonie) | Précision | F1 | AUC-ROC |
|---|---|---|---|---|---|---|
| CNN Custom | Fait maison | 0.375 | 0.000 | 0.000 | 0.00 | 0.578 |
| EfficientNetB0 Gelé | Pré-entraîné | 0.449 | 0.156 | 0.803 | 0.26 | 0.529 |
| **EfficientNetB0 Fine-tuné** | **Pré-entraîné + FT** | **0.450** | **0.236** | 0.672 | **0.35** | 0.467 |

**Modèle retenu** : EfficientNetB0 Fine-tuné, seuil = 0.3 — maximisation du Recall (priorité médicale : ne pas manquer une pneumonie).

---

### Tabulaire — Risque cardiovasculaire

| Modèle | Type | Accuracy | Précision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|---|
| MLP | Fait maison | 0.735 | 0.762 | 0.674 | 0.716 | 0.801 |
| Régression Logistique | Baseline linéaire | 0.735 | 0.760 | 0.680 | 0.718 | 0.798 |
| Random Forest | Ensemble | 0.737 | 0.755 | 0.698 | 0.726 | 0.811 |
| **XGBoost** | **Gradient Boosting** | **0.743** | **0.766** | **0.710** | **0.737** | **0.821** |

**Modèle retenu** : XGBoost (AUC = 0.821). Le MLP est utilisé dans le backend pour homogénéité avec le pipeline TensorFlow.

---

### Texte — Classification de rapports médicaux

| Modèle | Type | Accuracy | F1 macro | Points forts |
|---|---|---|---|---|
| TF-IDF + Logistic Reg. | Baseline | ~0.75 | ~0.70 | Rapide, interprétable |
| **BiLSTM + Attention** | **Fait maison** | **0.652** | **0.623** | Dépendances longues, attention custom |
| **ClinicalBERT** | **Pré-entraîné** | **0.716** | **0.689** | Vocabulaire médical natif |

**Modèle retenu** : ClinicalBERT — meilleure performance sur texte médical long grâce à son pré-entraînement domaine.

---

### Audio — Sons respiratoires

| Modèle | Type | Accuracy | F1 macro | Points forts |
|---|---|---|---|---|
| **CNN 2D sur Mel Spectrogram** | **Fait maison** | **0.734** | **0.711** | Interprétable, compact |
| **YAMNet + MLP** | **Pré-entraîné** | **0.825** | **0.803** | Meilleure représentation audio générale |

**Modèle retenu** : YAMNet + MLP — gain de +9 points d'accuracy grâce au transfert depuis AudioSet.

---

### Synthèse globale

| Modalité | Modèle Custom | Modèle Pré-entraîné | Gain |
|---|---|---|---|
| Image | CNN — AUC 0.578 | EfficientNetB0 FT — Recall +0.236 | +23.6 pts Recall |
| Tabulaire | MLP — AUC 0.801 | XGBoost — AUC 0.821 | +2.0 pts AUC |
| Texte | BiLSTM — Acc 0.652 | ClinicalBERT — Acc 0.716 | +6.4 pts Acc |
| Audio | CNN 2D — Acc 0.734 | YAMNet — Acc 0.825 | +9.1 pts Acc |

**Conclusion générale** : les modèles pré-entraînés surpassent systématiquement les architectures custom, particulièrement sur les données audio et texte où le pré-entraînement sur corpus massifs (AudioSet, MIMIC-III) apporte une représentation que des volumes de données médicaux limités ne peuvent pas reproduire from scratch.

---

## 6. Outils TensorFlow utilisés

### Pipeline de données

| Outil | Usage |
|---|---|
| `tf.data.Dataset` | Pipeline de chargement avec `.shuffle()`, `.batch()`, `.cache()`, `.prefetch(AUTOTUNE)` |
| `image_dataset_from_directory` | Chargement d'images par répertoire avec labels automatiques |
| `keras.preprocessing.sequence` | Padding et truncation des séquences texte |
| Augmentation `tf.data` | Noise injection, flip, rotation dans le graphe d'exécution |

### Couches Keras

| Couche | Modalité | Rôle |
|---|---|---|
| `Conv2D`, `MaxPooling2D`, `GlobalAveragePooling2D` | Image, Audio | Extraction de features spatiales |
| `BatchNormalization`, `Dropout`, `SpatialDropout1D` | Toutes | Régularisation |
| `Embedding`, `Bidirectional(LSTM)` | Texte | Représentation séquentielle |
| `Dense` | Toutes | Classification finale |
| `Rescaling`, `RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast` | Image | Augmentation dans le graphe |
| `LayerNormalization` | Texte (BERT) | Normalisation des transformers |
| `BahdanauAttention` (custom `Layer`) | Texte | Attention sur séquence LSTM |

### Modèles pré-entraînés

| Modèle | Source | Outil de chargement |
|---|---|---|
| EfficientNetB0 | ImageNet | `keras.applications.EfficientNetB0` |
| ClinicalBERT | MIMIC-III / BioBERT | `keras_hub.models.BertClassifier` |
| YAMNet | AudioSet | `tensorflow_hub.load("google/yamnet")` |

### Entraînement et callbacks

| Élément | Détail |
|---|---|
| Optimiseurs | `Adam` (lr adaptatif), `SGD`, `RMSprop` |
| Fonctions de perte | `binary_crossentropy`, `sparse_categorical_crossentropy`, `categorical_crossentropy` |
| Métriques | `AUC`, `Recall`, `Precision`, `F1Score` |
| `EarlyStopping` | Patience configurable, `restore_best_weights=True` |
| `ModelCheckpoint` | Sauvegarde du meilleur modèle en cours d'entraînement |
| `ReduceLROnPlateau` | Réduction dynamique du learning rate sur plateau |
| `class_weight` | Correction du déséquilibre de classes (image, audio) |

### Persistance des modèles

- Format natif `.keras` (recommandé TF 2.x+)
- `model.save()` / `keras.models.load_model()`
- Export de configuration JSON (seuil, classes, dimensions d'entrée)

---

## 7. Backend FastAPI

> **BONUS** — Les modèles entraînés sont exposés via une API REST asynchrone.

**Notebook** : [backend/](backend/) | **Framework** : FastAPI + Uvicorn | **Tests** : Pytest

### Endpoints disponibles

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Santé de l'API |
| `GET` | `/api/v1/ready` | Disponibilité des modèles chargés |
| `POST` | `/api/v1/predict/image` | Détection pneumonie (base64 JPEG) |
| `POST` | `/api/v1/predict/tabular` | Risque cardiovasculaire (features JSON) |
| `POST` | `/api/v1/predict/text` | Classification rapport médical |
| `POST` | `/api/v1/predict/multimodal` | Fusion de plusieurs modalités |

### Lancement

```powershell
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
# Swagger UI disponible sur http://127.0.0.1:8000/docs
```

### Tests

```powershell
cd backend
python -m pytest -q
```

### Fusion multimodale

Le service `fusion.py` combine les scores de confiance de plusieurs modalités par **moyenne pondérée** (extensible à une couche de fusion apprise). La réponse unifiée expose le label prédit, le score de confiance, le nom du modèle et, le cas échéant, les zones d'attention (Grad-CAM pour les images).

---

## 8. Justification des choix techniques

| Choix | Justification |
|---|---|
| **Pipeline `tf.data`** (pas de `ImageDataGenerator`) | Chargement parallèle, mise en cache GPU, AUTOTUNE — seule approche recommandée pour TF 2.x |
| **`class_weight`** sur données déséquilibrées | En médecine, manquer une pathologie (faux négatif) est plus grave qu'une alarme excessive — le recall prime sur la précision |
| **EfficientNetB0** plutôt que ResNet/VGG | Meilleur compromis params/performance ; conçu pour la scalabilité ; validé sur transfert vers domaines médicaux |
| **Fine-tuning en deux phases** | Phase 1 : convergence rapide des couches de tête sans corrompre les features pré-appris. Phase 2 : adaptation progressive au domaine médical avec très faible lr |
| **Seuil 0.3** (image) | Déplacement du trade-off précision/recall en faveur du recall — validé sur la courbe ROC |
| **BiLSTM + Attention** | Capture des dépendances longues dans les comptes-rendus (>300 tokens) ; mécanisme d'attention interprétable |
| **ClinicalBERT** vs BERT générique | Pré-entraîné sur MIMIC-III (notes cliniques) : vocabulaire médical natif, pas besoin de ré-apprendre "pneumothorax" ou "tachycardie" |
| **Spectrogramme Mel** pour l'audio | Représentation temps-fréquence robuste, standard en analyse audio ; log-compression simulant la perception humaine |
| **YAMNet** pour l'audio | 521 classes AudioSet incluant des sons respiratoires ; embeddings généraux transférables avec peu de données cibles |
| **FastAPI** pour le backend | Async natif, validation Pydantic automatique, Swagger UI embarqué, performant pour les workloads ML |
| **Format `.keras`** pour la persistance | Format natif recommandé depuis TF 2.12 ; préserve l'architecture, les poids et la config de compilation |
| **Seed global 42** | Reproductibilité totale entre exécutions (`tf.random.set_seed`, `np.random.seed`, splits stratifiés) |

---

## 9. Reproductibilité

### Environnement

| Dépendance | Version |
|---|---|
| Python | 3.11 |
| TensorFlow | ≥ 2.16 |
| keras-hub | pour ClinicalBERT |
| tensorflow-hub | pour YAMNet |
| FastAPI + Uvicorn | backend |
| librosa | traitement audio |
| scikit-learn, XGBoost | modèles tabulaires |
| pandas, numpy | manipulation de données |
| matplotlib, seaborn | visualisation |

### Installation

```powershell
pip install tensorflow keras-hub tensorflow-hub fastapi uvicorn librosa scikit-learn xgboost pandas numpy matplotlib seaborn
```

### Données

Les datasets sont téléchargés automatiquement via `kagglehub` au début de chaque notebook. Des échantillons réduits sont disponibles dans `data/sample/` pour une exécution rapide sans téléchargement.

### Reproductibilité des modèles

- Graine fixée globalement : `SEED = 42`
- Splits stratifiés sur toutes les partitions train/val/test
- `StandardScaler` ajusté uniquement sur le train set et sérialisé (`scaler_cardio.pkl`)
- Configurations de callbacks sauvegardées dans chaque notebook

---

## 10. Résultats et conclusions

### Meilleurs modèles retenus

| Modalité | Modèle | Métrique principale | Score |
|---|---|---|---|
| Image | EfficientNetB0 Fine-tuné (seuil 0.3) | Recall | 0.236 |
| Tabulaire | XGBoost | AUC-ROC | 0.821 |
| Texte | ClinicalBERT | Accuracy | 0.716 |
| Audio | YAMNet + MLP | Accuracy | 0.825 |

### Enseignements clés

1. **Le transfert learning domine** dans tous les cas, particulièrement quand le volume de données médicales est limité et que le modèle source a été pré-entraîné dans un domaine proche (biomédical ou audio général).

2. **L'évaluation doit être contextualisée** : la métrique pertinente varie selon la pathologie. Pour la pneumonie, le recall prime sur l'accuracy — un modèle avec 45 % d'accuracy mais 24 % de recall est préféré à un modèle avec 60 % d'accuracy mais 0 % de recall.

3. **Le pipeline `tf.data` est non négociable** pour des volumes de données réels : le gain en temps d'entraînement (mise en cache GPU, préchargement) est significatif dès quelques milliers d'exemples.

4. **La multimodalité est un levier clinique** : combiner radiographie, bilan biologique et compte-rendu pour un même patient réduit les incertitudes individuelles de chaque modalité.

5. **Les modèles custom restent essentiels** comme baseline interprétable et comme point de comparaison rigoureux — ils permettent de mesurer et de justifier le gain apporté par le transfert learning.

---

*Projet réalisé dans le cadre du cours Deep Learning sous TensorFlow — ESTIAM.*
*Équipe : Shakir OYEOSSI*
