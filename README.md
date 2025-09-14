
# 📊 Credit Risk Prediction in P2P Lending

## 📄 Overview
This project implements a machine learning framework for **credit risk prediction** in Peer-to-Peer (P2P) lending. Using the **Bondora loan dataset (2012–2016)**, the system predicts whether a borrower is likely to **default** or **repay**.  
The model assists lenders in making **data-driven lending decisions** and minimizing risk.

---

## ⚙ Features
- **Data Preprocessing**: Cleaning, handling missing values, encoding categorical features.
- **Class Imbalance Handling**: Applied oversampling & undersampling techniques to balance default vs. non-default loans.
- **Feature Selection**: Hybrid approach using model-based feature importance and SHAP explainability.
- **Model Training**: Comparative analysis of multiple ML models.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, AUC-ROC.

---

## 🛠 Technologies Used
- **Languages**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, CatBoost, Matplotlib, SHAP  
- **Tools**: Jupyter Notebook, Google Colab  

---

## 📁 Project Structure
```

Credit\_Risk\_Prediction/
│
├── data/                 # Dataset files (Bondora loan dataset)
├── notebooks/            # Jupyter Notebooks for each stage
│   ├── 1\_data\_preprocessing.ipynb
│   ├── 2\_feature\_engineering.ipynb
│   ├── 3\_model\_training.ipynb
│   └── 4\_evaluation.ipynb
├── src/                  # Source code scripts
│   ├── preprocessing.py
│   ├── models.py
│   ├── utils.py
│   └── feature\_selection.py
├── results/              # Evaluation results, plots, SHAP explainability graphs
└── README.md             # Project documentation

````

---

## 🚀 Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/your-username/Credit_Risk_Prediction.git
cd Credit_Risk_Prediction
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Notebooks

* **Data Preprocessing** → notebooks/1\_data\_preprocessing.ipynb
* **Feature Engineering & Selection** → notebooks/2\_feature\_engineering.ipynb
* **Model Training** → notebooks/3\_model\_training.ipynb
* **Evaluation** → notebooks/4\_evaluation.ipynb

---

## 📊 Model Evaluation

### Models Compared

* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost

### Best Model

* **Random Forest with Hybrid Feature Selection**
* **AUC-ROC**: 0.9131
* **F1-Score**: 0.8213

---

## 📈 Results & Visualizations

* **Class Distribution** (default vs. non-default)
* **SHAP Values** for feature importance & explainability
* **Confusion Matrix** for model predictions
* **ROC Curve** to measure classification performance

---

## 🔄 Future Improvements

* Experiment with **deep learning models** (e.g., LSTM, TabNet).
* Apply **cost-sensitive learning** to reduce false negatives.
* Deploy as an **API** or **web dashboard** for real-time credit scoring.
