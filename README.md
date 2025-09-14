
# 📊 Credit Risk Prediction in P2P Lending

## 📄 Overview
This project focuses on predicting **credit risk in Peer-to-Peer (P2P) lending** using machine learning techniques.  
It uses the **Bondora loan dataset (2012–2016)** to classify whether a borrower is likely to **default** or **repay**.  
The aim is to help investors and lending platforms make better data-driven decisions.

---

## ⚙ Features
- Cleaned and preprocessed raw loan data.  
- Applied **class balancing** (oversampling/undersampling).  
- Hybrid feature selection (SHAP + model feature importance).  
- Compared multiple ML models (XGBoost, LightGBM, CatBoost, Random Forest).  
- Generated **AUC curves, confusion matrices, and feature importance tables**.  
- Summarized results in **evaluation metrics tables**.

---

## 🛠 Technologies
- **Language**: Python 3.9+  
- **Libraries**:  
  - Data: Pandas, NumPy  
  - ML Models: Scikit-learn, XGBoost, LightGBM, CatBoost  
  - Visualization: Matplotlib, Seaborn, SHAP  
- **Tools**: Jupyter Notebook, VS Code  

---

## 📁 Folder Structure
```

Credit-Risk-Prediction/
│
├── data/                        # Raw / preprocessed datasets
│
├── outputs/                     # Results & evaluation outputs
│   ├── Auc\_curves/              # ROC/AUC curves for models
│   ├── confusion\_matrices/      # Confusion matrix plots
│   ├── evaluation\_metrices\_table/ # Tables with accuracy, F1, AUC etc.
│   └── top\_15\_features/         # SHAP / importance plots
│
├── src/                         # Source code for pipeline
│   ├── evaluate.py              # Model evaluation scripts
│   ├── preprocessing.py         # Data preprocessing pipeline
│   ├── re                       # (check if placeholder, can remove)
│   └── utils.py                 # Helper functions
│
├── Train/                       # Training-related scripts
│   └── (model training files)
│
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Ignored files

````

---

## 🚀 Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/your-username/Credit-Risk-Prediction.git
cd Credit-Risk-Prediction
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Training & Evaluation

* **Preprocessing**

  ```bash
  python src/preprocessing.py
  ```
* **Model Training (inside Train/)**

  ```bash
  python Train/train_model.py
  ```
* **Evaluation**

  ```bash
  python src/evaluate.py
  ```

Outputs (confusion matrices, AUC curves, metrics tables, SHAP features) will be saved in the **`outputs/`** folder.

---

## 📊 Model Evaluation

### Models Compared

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* LightGBM
* CatBoost

### Example Metrics

* **Best Model**: Random Forest with hybrid feature selection
* **AUC**: 0.91
* **F1-score**: 0.82

---

## 📈 Results & Visualizations

* Confusion Matrices → `outputs/confusion_matrices/`
* ROC / AUC Curves → `outputs/Auc_curves/`
* Feature Importance (Top 15) → `outputs/top_15_features/`
* Evaluation Summary → `outputs/evaluation_metrices_table/`

---

## 🔮 Future Work

* Try **deep learning models** (TabNet, Autoencoders).
* Add **cost-sensitive learning** to handle imbalanced data.
* Deploy as a **FastAPI/Flask service** for real-time scoring.




👉 Do you also want me to **regenerate `requirements.txt`** (scikit-learn, xgboost, catboost, shap, etc.) so anyone can set it up in one command?
```
