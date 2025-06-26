
---

### ✅ Final `README.md` (No License Section)

```markdown
# 🧠 Credit Risk Prediction using SHAP and Hybrid Feature Selection

This project presents an end-to-end machine learning pipeline to predict credit risk using ensemble models and explainable AI (SHAP).  
It is part of the B.Tech major project titled:

**“Improving Default Prediction in Peer-to-Peer Lending with Feature Selection and Explainable ML Models”**  
📍 Department of Computer Science, NIT Karnataka

---

## 📁 Project Structure

```

CREDIT-RISK/
├── data/
│   └── Trainset.csv
├── models/
├── notebooks/
├── outputs/
│   ├── Auc\_curves/
│   ├── confusion\_matrix/
│   ├── evaluation\_metrics/
│   └── top\_15\_features/
├── src/
│   ├── Train/
│   │   ├── catboost\_basic.py
│   │   ├── catboost\_shap.py
│   │   ├── catboost\_shap\_union.py
│   │   ├── decision\_tree\_basic.py
│   │   ├── decision\_tree\_shap.py
│   │   ├── decision\_tree\_shap\_union.py
│   │   ├── gradient\_boost\_basic.py
│   │   ├── gradient\_boost\_shap.py
│   │   ├── gradient\_boost\_shap\_union.py
│   │   ├── lightgbm\_basic.py
│   │   ├── lightgbm\_shap.py
│   │   ├── lightgbm\_shap\_union.py
│   │   ├── random\_forest\_basic.py
│   │   ├── random\_forest\_shap.py
│   │   ├── random\_forest\_shap\_union.py
│   │   ├── xgboost\_basic.py
│   │   ├── xgboost\_shap.py
│   │   ├── xgboost\_shap\_union.py
│   ├── evaluate.py
│   ├── preprocessing.py
│   └── utils.py
├── README.md
└── requirements.txt

````

---

## 🚀 Highlights

- ✅ SHAP-based interpretability for every model
- ✅ Hybrid feature selection: SHAP + model-based
- 🔁 Models implemented: XGBoost, LightGBM, Random Forest, CatBoost, Gradient Boost, Decision Tree
- ⚡ Each model trained in 3 versions: `basic`, `shap`, `shap + union`
- 📊 Outputs include: AUC curves, confusion matrices, feature importance

---

## 💻 How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
````

### 2️⃣ Run Preprocessing

```bash
python src/preprocessing.py
```

### 3️⃣ Train Any Model (Examples)

```bash
# XGBoost
python src/Train/xgboost_basic.py
python src/Train/xgboost_shap.py
python src/Train/xgboost_shap_union.py

# LightGBM
python src/Train/lightgbm_basic.py
python src/Train/lightgbm_shap.py
python src/Train/lightgbm_shap_union.py

# Random Forest
python src/Train/random_forest_basic.py
python src/Train/random_forest_shap.py
python src/Train/random_forest_shap_union.py

# CatBoost
python src/Train/catboost_basic.py
python src/Train/catboost_shap.py
python src/Train/catboost_shap_union.py

# Gradient Boost
python src/Train/gradient_boost_basic.py
python src/Train/gradient_boost_shap.py
python src/Train/gradient_boost_shap_union.py

# Decision Tree
python src/Train/decision_tree_basic.py
python src/Train/decision_tree_shap.py
python src/Train/decision_tree_shap_union.py
```

### 4️⃣ Evaluate Results

```bash
python src/evaluate.py
```

### 5️⃣ Check Outputs

All results are saved inside the `outputs/` folder:

* Confusion matrices
* AUC curves
* SHAP plots
* Evaluation metric tables
* Top 15 features (text)

---

## 🧪 Output Samples

### 📊 Evaluation Table

![Evaluation Table](outputs/evaluation_metrics/evaluation_metrics_table.png)

### 🌀 Confusion Matrix Example

![Confusion Matrix](outputs/confusion_matrix/xgboost_confusion_matrix.png)

---

## 📚 Requirements

Common packages used (in `requirements.txt`):

* `scikit-learn`
* `xgboost`, `lightgbm`, `catboost`
* `shap`
* `matplotlib`, `seaborn`, `pandas`, `numpy`

---

## 🙋‍♂️ Author

**Prince Kumar Chaudhary**
📍 B.Tech – Computer Science, NITK Surathkal
