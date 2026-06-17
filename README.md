<div align="center">

# 🚗 CO₂ Emission Prediction & Classification

**Predict vehicle CO₂ emissions and classify them into emission bands using machine learning.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikitlearn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat-square&logo=kaggle)

</div>

---

## 📌 Overview

This project tackles CO₂ emission analysis from two angles:

| Task | Goal | Best Model |
|---|---|---|
| 🔢 **Regression** | Predict exact CO₂ value (g/km) | Gradient Boosting |
| 🏷️ **Classification** | Assign emission band (Low / Medium / High) | Random Forest |

A unified prediction pipeline takes raw vehicle specs and returns both a numeric CO₂ estimate and an emission band label in a single call.

---

## 📊 Dataset

> **Source:** [Canadian Vehicle CO₂ Emissions Dataset — Kaggle](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles)

- Trained and evaluated on a **filtered subset of 679 vehicle records**
- Full dataset included in the repository for reference and reproducibility
- 12 columns covering make, model, class, engine specs, fuel type, fuel consumption, and CO₂ emissions

### Features Used

| Feature | Description |
|---|---|
| `VEHICLE CLASS` | Category of vehicle (e.g., Compact, SUV) |
| `FUEL` | Fuel type (X = Regular, Z = Premium, …) |
| `TRANSMISSION` | Transmission type and number of gears |
| `ENGINE_SIZE` | Engine displacement in litres |
| `CYLINDERS` | Number of engine cylinders |
| `FUEL_CONSUMPTION*` | Combined fuel consumption (L/100 km) |
| `CO2_EMISSIONS` | 🎯 Regression target — CO₂ in g/km |
| `Emission_Band` | 🎯 Classification target — Low / Medium / High |

---

## 🗂️ Project Structure

```
CO2-Emission-Prediction-Classification/
│
├── Code/
│   └── ML_FinalProject.ipynb               # Main Jupyter notebook
│
├── Dataset/
│   └── CO2 Emissions_Canada.csv            # Full dataset
│
├── Images/
│   ├── co2_distribution.jpeg
│   ├── model_comparison.jpeg
│   ├── actual_vs_predicted.jpeg
│   ├── feature_importance.jpeg
│   └── confusion_matrix.jpeg
│
├── MachineLearningFinalProject_Report.pdf  # Full project report
├── ML_FinalProject.pdf                     # Notebook export (PDF)
└── README.md
```

---

## ⚙️ Methodology

### Data Preprocessing
- Removed duplicates and null values
- Dropped unnamed/index columns
- Selected relevant feature columns for ML
- One-hot encoded categorical features (`VEHICLE CLASS`, `FUEL`, `TRANSMISSION`) via `ColumnTransformer`

### 🔢 Regression — Predicting CO₂ (g/km)

Seven models compared on **MAE**, **RMSE**, and **R²**:

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| K-Nearest Neighbors | |
| Decision Tree | |
| AdaBoost | |
| Bagging | |
| Random Forest | |
| **Gradient Boosting** ✅ | **Best — lowest RMSE** |

The Gradient Boosting model was wrapped in a full `sklearn` Pipeline for clean, reproducible inference.

### 🏷️ Classification — Emission Band (Low / Medium / High)

Five classifiers compared by **accuracy**:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline |
| Decision Tree | |
| SVM (RBF kernel) | |
| KNN (k=5) | |
| **Random Forest** ✅ | **Best — highest accuracy** |

The Random Forest classifier was also packaged into a Pipeline, with a full confusion matrix evaluated across all three emission band classes.

---

## 📈 Visualisations

<table>
  <tr>
    <td align="center"><b>CO₂ Distribution</b></td>
    <td align="center"><b>Model Comparison (RMSE)</b></td>
  </tr>
  <tr>
    <td><img src="Images/co2_distribution.jpeg" width="380"/></td>
    <td><img src="Images/model_comparison.jpeg" width="380"/></td>
  </tr>
  <tr>
    <td align="center"><b>Actual vs Predicted</b></td>
    <td align="center"><b>Feature Importance</b></td>
  </tr>
  <tr>
    <td><img src="Images/actual_vs_predicted.jpeg" width="380"/></td>
    <td><img src="Images/feature_importance.jpeg" width="380"/></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><b>Confusion Matrix</b></td>
  </tr>
  <tr>
    <td colspan="2" align="center"><img src="Images/confusion_matrix.jpeg" width="380"/></td>
  </tr>
</table>

---

## 🚀 Usage

### Installation

```bash
git clone https://github.com/your-username/CO2-Emission-Prediction-Classification.git
cd CO2-Emission-Prediction-Classification
pip install -r requirements.txt
```

Then open `Code/ML_FinalProject.ipynb` in Jupyter and update the CSV path in the data loading cell to match your local setup.

### Predict CO₂ for a New Vehicle

```python
new_car = {
    'VEHICLE CLASS': 'COMPACT',
    'FUEL': 'X',
    'TRANSMISSION': 'A4',
    'ENGINE_SIZE': 1.8,
    'CYLINDERS': 4,
    'FUEL_CONSUMPTION*': 7.5
}

pred_co2, pred_band = predict_co2_and_band_pipeline(
    car_dict=new_car,
    reg_pipeline=gb_pipeline,   # Gradient Boosting
    clf_pipeline=rf_pipeline    # Random Forest
)

print(f"Predicted CO₂:  {pred_co2:.1f} g/km")
print(f"Predicted Band: {pred_band}")
```

### Predict for an Existing Record

```python
idx = 0  # any valid row index in df
existing_car_df = df.loc[[idx], feature_cols]

pred_co2  = gb_pipeline.predict(existing_car_df)[0]
pred_band = rf_pipeline.predict(existing_car_df)[0]

print(f"True CO₂: {df.loc[idx, 'CO2_EMISSIONS']} | Predicted: {pred_co2:.1f}")
print(f"True Band: {df.loc[idx, 'Emission_Band']} | Predicted: {pred_band}")
```

---

## 🛠️ Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset sourced from the [Canadian Vehicle CO₂ Emissions Dataset](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles) on Kaggle
- Built with [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), and [matplotlib](https://matplotlib.org/)

