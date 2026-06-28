# Workflow-CI — MSML Kriteria 3

[![CI - Train Model and Build Docker Image](https://github.com/arvamadax/Workflow-CI/actions/workflows/ci.yml/badge.svg)](https://github.com/arvamadax/Workflow-CI/actions/workflows/ci.yml)

Repository ini berisi **MLflow Project** untuk training model Random Forest dan **GitHub Actions CI** yang otomatis melakukan re-training, upload artefak, build & push Docker image ke Docker Hub.

Bagian dari submission Proyek Akhir **Membangun Sistem Machine Learning (MSML)** Dicoding.

## Dataset & Model

- **Dataset:** Telco Customer Churn (7.043 sampel, binary classification)
- **Model:** Random Forest Classifier dengan hyperparameter hasil tuning Kriteria 2:

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 30 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 4 |
| `max_features` | sqrt |
| `class_weight` | balanced |
| `random_state` | 42 |

### Hasil Training (Deterministik)

| Metric | Train | Test |
|---|---|---|
| Accuracy | 0.86 | 0.78 |
| F1 Score | 0.74 | 0.62 |
| ROC-AUC | 0.91 | 0.84 |

## Struktur Repository

```
Workflow-CI/
├── .github/workflows/
│   └── ci.yml                       # GitHub Actions: train → artifact → Docker
├── MLProject/
│   ├── MLProject                    # MLflow Project definition
│   ├── conda.yaml                   # conda environment
│   ├── requirements.txt
│   ├── modelling.py                 # training script (best params dari K2)
│   └── namadataset_preprocessing/
│       ├── train.csv
│       └── test.csv
├── Docker_Hub.txt                   # tautan ke Docker Hub image
├── README.md
└── .gitignore
```

## Cara Menjalankan Lokal

### Setup
```bash
pip install -r MLProject/requirements.txt
```

### Run via MLflow Project (recommended)
```bash
cd MLProject
mlflow run . --env-manager=local
```

### Run langsung (testing cepat)
```bash
cd MLProject
python modelling.py
```

### Override hyperparameter
```bash
cd MLProject
mlflow run . --env-manager=local -P n_estimators=300 -P max_depth=20
```

## GitHub Actions CI

Workflow `ci.yml` ter-trigger otomatis pada:

- Push ke `main` yang mengubah `MLProject/` atau workflow
- Pull request ke `main`
- Manual via tab Actions (`workflow_dispatch`)

### Yang Dilakukan CI

1. **Setup environment** — Python 3.12.7, install MLflow + dependencies
2. **Training** — jalankan `mlflow run` → log metrik + 5 artefak ke MLflow
3. **Upload artifact** — semua isi `mlruns/` di-upload sebagai GitHub artifact (retention 30 hari)
4. **Build Docker image** — `mlflow models build-docker` dengan model artifact
5. **Push ke Docker Hub** — tag dengan `latest`, `<commit-sha>`, `run-<n>`

### GitHub Secrets yang Dibutuhkan

Set di **Settings → Secrets and variables → Actions**:

| Secret Name | Value |
|---|---|
| `DOCKER_HUB_USERNAME` | username Docker Hub |
| `DOCKER_HUB_TOKEN` | personal access token (bukan password) |

## Mapping ke Rubrik

| Level | Requirement | Implementasi |
|---|---|---|
| **Basic (2)** | MLflow Project + workflow CI training | `MLProject/` + `ci.yml` job training |
| **Skilled (3)** | + simpan artefak ke repo / Drive / LFS | `actions/upload-artifact@v4` |
| **Advanced (4)** | + build Docker Image ke Docker Hub via `mlflow build-docker` | Steps `Build Docker image with mlflow` + `Tag & push` |

## Docker Image

Image yang dihasilkan adalah **MLflow model server** yang menjalankan model di port 8080.

### Pull & Run Image
```bash
docker pull arvamadax/telco-churn-mlflow:latest

# Serve model di port 5001 (host) → 8080 (container)
docker run -p 5001:8080 arvamadax/telco-churn-mlflow:latest
```

### Test Inference
```bash
curl http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": [...], "data": [[...]]}}'
```

Tautan Docker Hub tersedia di file [`Docker_Hub.txt`](./Docker_Hub.txt).

## Penulis

**Arva Mada Jayastu** — Teknik Komputer, FILKOM Universitas Brawijaya
NIM 255150300111053 · GitHub [@arvamadax](https://github.com/arvamadax)
