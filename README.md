# iris-random-forest
A Random Forest classifier for Iris flower species prediction — includes data preprocessing, model evaluation, feature importance analysis, and model export using scikit-learn.
# 🌸 Iris Flower Classification — Random Forest

A machine learning project that classifies iris flowers into three species (**Setosa**, **Versicolor**, **Virginica**) using a Random Forest classifier built with scikit-learn.

---

## 📌 Project Overview

This notebook walks through a complete ML pipeline — from data loading and cleaning to model training, evaluation, and saving — applied to the classic Iris dataset.

---

## 📁 Repository Structure

```
├── random_forest_iris.ipynb   # Main Jupyter notebook
├── iris.csv                   # Dataset
├── iris_model.joblib          # Saved model + scaler (generated after running notebook)
└── README.md
```

---

## 🔍 Dataset

- **Source:** `iris.csv`
- **Records:** 150 (147 after removing 3 duplicates)
- **Features:**
  | Feature | Description |
  |---|---|
  | `sepal_length` | Sepal length in cm |
  | `sepal_width` | Sepal width in cm |
  | `petal_length` | Petal length in cm |
  | `petal_width` | Petal width in cm |
- **Target:** `species` — one of `setosa`, `versicolor`, `virginica`

---

## 🧪 Workflow

1. **Data Loading** — Load `iris.csv` with pandas
2. **EDA** — Sample inspection, null checks, duplicate detection
3. **Preprocessing** — Drop duplicates, feature/target split, train-test split, `StandardScaler` normalization
4. **Modeling** — Train a `RandomForestClassifier`
5. **Evaluation** — Accuracy score, classification report, confusion matrix
6. **Visualization** — Feature importance bar chart, actual vs. predicted plot
7. **Model Saving** — Export model + scaler to `iris_model.joblib` using `joblib`

---

## 🛠️ Installation

```bash
git clone https://github.com/khanzalan/iris-random-forest.git
cd iris-random-forest
pip install -r requirements.txt
```

### Requirements

```
numpy
pandas
seaborn
matplotlib
scikit-learn
joblib
```

Or install directly:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn joblib
```

---

## 🚀 Usage

Launch the notebook:

```bash
jupyter notebook random_forest_iris.ipynb
```

Run all cells top to bottom. The trained model will be saved as `iris_model.joblib`.

To load the saved model later:

```python
import joblib

saved = joblib.load('iris_model.joblib')
model  = saved['model']
scaler = saved['scaler']
```

---

## 📊 Results

The Random Forest classifier achieves strong performance on the Iris dataset. Key evaluation outputs include:

- Accuracy score on the test set
- Per-class precision, recall, and F1-score
- Confusion matrix heatmap
- Feature importance ranking
- Actual vs. predicted comparison plot (first 20 test samples)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.
