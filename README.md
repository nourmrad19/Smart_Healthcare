Voici un **README complet** à coller dans `README.md` à la racine du projet, puis le point sur l’avancement.

---

## Contenu à mettre dans `README.md`

Dans VS Code : ouvrez (ou créez) `README.md` à la racine → sélectionnez tout → collez ceci → **Ctrl+S**.

```markdown
# Smart Healthcare Data Space

Plateforme permettant à **3 hôpitaux** de collaborer pour entraîner un modèle d’IA de prédiction de réadmission **sans partager leurs données médicales brutes**.

---

## 1. Objectif du projet

Les hôpitaux veulent un modèle plus robuste, mais ne peuvent pas centraliser les dossiers patients (RGPD / confidentialité).

**Solution :**
- Chaque hôpital garde ses données **localement**
- Entraînement **fédéré** (FedAvg) : seuls les **poids du modèle** sont agrégés
- Gouvernance type **Data Space / EDC** (catalogue & contrats sur les paramètres du modèle, pas sur les patients)
- API **FastAPI** + **Dashboard** pour suivre l’état et les métriques

```
Hospital 1 ─┐
Hospital 2 ─┼─→ Agrégation FedAvg → Modèle global
Hospital 3 ─┘
     │
     └─→ FastAPI + Dashboard
```
---- **L’idée du Federated Learning** ----
Au lieu d’envoyer les dossiers patients, chaque hôpital :

entraîne le modèle chez lui ;
envoie uniquement les poids (paramètres du réseau) ;
un coordinateur moyenne ces poids → modèle global ;
renvoie ce modèle global aux hôpitaux, et on recommence.

FedAvg = Federated Averaging = cette moyenne des modèles, pondérée par la taille des jeux de données locaux.
textHospital 1  →  poids W1  ─┐
Hospital 2  →  poids W2  ─┼─→  W_global = moyenne(W1, W2, W3)  →  modèle partagé
Hospital 3  →  poids W3  ─┘
---

## 2. Dataset utilisé

| Élément | Détail |
|---------|--------|
| **Nom** | UCI Diabetes 130-US Hospitals for Years 1999-2008 |
| **Source** | https://archive.ics.uci.edu/dataset/296 |
| **Fichier** | `diabetic_data.csv` (~19 MB) |
| **Tâche** | Classification binaire : réadmission sous 30 jours (`readmitted == "<30"`) |
| **Partition** | 3 hôpitaux via `StratifiedKFold` (pas de simple copie) |

### Comment fournir le dataset

#### Option A — PowerShell (Windows)

```powershell
cd C:\Users\mradn\Documents\smart-healthcare-dataspace
mkdir data\raw -ErrorAction SilentlyContinue
cd data\raw
Invoke-WebRequest -Uri "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip" -OutFile "dataset.zip"
python -c "import zipfile; zipfile.ZipFile('dataset.zip').extractall('.'); print('OK')"
cd ..\..
```

Vous devez avoir : `data\raw\diabetic_data.csv`

#### Option B — Navigateur

1. Télécharger :  
   https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip  
2. Extraire dans `data\raw\`  
3. Vérifier la présence de `diabetic_data.csv`

---

## 3. Stack technique

- **Python 3.12** + venv  
- **PyTorch** (CPU)  
- **FedAvg** (apprentissage fédéré)  
- **FastAPI** + **Uvicorn**  
- **Docker / Docker Compose**  
- **EDC** : simulation de gouvernance (catalogue / négociations) — les données patients ne transitent pas par EDC  
- Dataset public UCI (réel)

---

## 4. Structure du projet

```text
smart-healthcare-dataspace/
├── data/
│   ├── raw/diabetic_data.csv      # dataset UCI
│   ├── hospital_1|2|3/            # données locales par hôpital
│   └── processed/partition_meta.json
├── ml/
│   ├── preprocess.py              # nettoyage + partition
│   └── model.py                   # DiabetesNet (PyTorch)
├── federated_learning/
│   └── run_fedavg.py              # entraînement FedAvg
├── backend/
│   └── main.py                    # API FastAPI
├── dashboard/
│   └── index.html                 # interface web
├── results/                       # métriques + global_model.pt
├── tests/
│   └── test_pipeline.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 5. Installation (première fois)

```powershell
cd C:\Users\mradn\Documents\smart-healthcare-dataspace

# Environnement virtuel
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Dépendances
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Vérification :

```powershell
python -c "import torch, sklearn, fastapi; print('OK', torch.__version__)"
```

---

## 6. Lancer le projet (local, sans Docker)

### Étape 1 — Dataset
(voir section 2 si `data\raw\diabetic_data.csv` est absent)

### Étape 2 — Preprocessing (3 hôpitaux)

```powershell
python ml/preprocess.py
```

### Étape 3 — Entraînement fédéré

```powershell
python federated_learning/run_fedavg.py --rounds 5 --local-epochs 2
```

Exemple de résultat attendu :
- Accuracy ≈ **0.888**
- Loss ≈ **0.336**
- Durée ≈ **15–20 s** (CPU)

### Étape 4 — Tests

```powershell
python tests/test_pipeline.py
```

Attendu : `6/6 passed`

### Étape 5 — API + Dashboard

```powershell
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Puis dans le navigateur :

| URL | Description |
|-----|-------------|
| **http://localhost:8000** | **Dashboard** (métriques, graphiques, hôpitaux, EDC) |
| **http://localhost:8000/docs** | Documentation interactive de l’API (Swagger) |
| http://localhost:8000/health | Santé du service |
| http://localhost:8000/training/status | Statut de l’entraînement |
| http://localhost:8000/hospitals | État des 3 hôpitaux |
| http://localhost:8000/edc/status | Connecteurs EDC (simulation) |
| http://localhost:8000/system/status | Vue système + privacy |

> **Important :** utiliser `http://localhost:8000` et **non** `http://0.0.0.0:8000`.

---

## 7. Lancer avec Docker

Prérequis : **Docker Desktop** démarré.

```powershell
cd C:\Users\mradn\Documents\smart-healthcare-dataspace
docker compose up --build
```

Dashboard : **http://localhost:8000**

Arrêt : `Ctrl+C` dans le terminal.

---

## 8. Endpoints API

| Méthode | Chemin | Rôle |
|---------|--------|------|
| GET | `/health` | Liveness |
| GET | `/hospitals` | 3 hôpitaux + métriques locales |
| GET | `/training/status` | Statut FL |
| GET | `/training/metrics` | Accuracy / loss par round |
| GET | `/edc/status` | Connecteurs + catalogue + négociations |
| GET | `/edc/catalog` | Assets (paramètres modèle uniquement) |
| GET | `/edc/negotiations` | Contrats simulés |
| GET | `/system/status` | Vue d’ensemble + garanties privacy |
| GET | `/` | Dashboard HTML |

---

## 9. Privacy (garanties)

| Partagé | Non partagé |
|---------|-------------|
| Poids du modèle (tenseurs) | Dossiers patients (CSV / NPZ) |
| Métriques agrégées (accuracy, loss) | Features individuelles, IDs |
| Métadonnées de contrat EDC | Toute donnée nominative / PII |

Les fichiers `data/hospital_1|2|3/` restent locaux à chaque “hôpital”.  
EDC ne centralise **pas** les données médicales : uniquement la gouvernance de l’échange de **paramètres de modèle**.

---

## 10. Résultats obtenus (exemple réel)

```text
Round 1/5 | loss≈0.337 | acc≈0.888
Round 5/5 | loss≈0.336 | acc≈0.888
Temps total ≈ 16 s (CPU)
```

Fichiers générés :
- `results/training_history.json`
- `results/final_summary.json`
- `results/global_model.pt`

---

## 11. Limites assumées (transparence)

- **EDC** : simulation de catalogue / négociation dans FastAPI (pas le runtime Java Eclipse complet).
- **Flower** : algorithme FedAvg en processus unique (équivalent fonctionnel ; pas 3 processus client Flower séparés).
- **Kubernetes** : non déployé dans cette version (Docker Compose couvre le déploiement conteneurisé).

Ces choix permettent un prototype **exécutable et démontrable** sur machine standard.

---

## 12. Citation dataset

Strack et al., *Impact of HbA1c Measurement on Hospital Readmission Rates*, BioMed Research International, 2014.  
UCI ML Repository : https://archive.ics.uci.edu/dataset/296
```

---

## Vous en êtes où dans le projet ?

### Terminé (cœur du brief)

| Élément | Statut |
|---------|--------|
| Dataset UCI réel | Fait |
| Partition 3 hôpitaux | Fait |
| Modèle PyTorch | Fait |
| FedAvg + métriques **réelles** | Fait |
| FastAPI (tous les endpoints demandés) | Fait |
| Dashboard | Fait |
| Docker Compose | Fait |
| Tests 6/6 | Fait |
| README | À coller maintenant |


---

## Commandes rapides (rappel)

```powershell
.\.venv\Scripts\Activate.ps1
python ml/preprocess.py
python federated_learning/run_fedavg.py --rounds 5
python tests/test_pipeline.py
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Dashboard : **http://localhost:8000**  
API docs : **http://localhost:8000/docs**

