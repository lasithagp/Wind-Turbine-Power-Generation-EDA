# 🌬️ Wind Turbine Power Generation: Exploratory Data Analysis (EDA)

## 📌 Project Overview

This project presents an in-depth Exploratory Data Analysis (EDA) of wind turbine power generation [dataset](https://www.kaggle.com/datasets/sudhanvahg/wind-turbine-power-generation-forecasting/) obtained from kaggle. 
[Part 1](https://nbviewer.org/github/lasithagp/Wind-Turbine-Power-Generation-EDA/blob/main/notebooks/Wind-Turbine-Power-Generation-EDA.ipynb) aims to uncover the key patterns, trends, and relationships between wind speed, direction, and generated power. 
Includes visual analytics, domain insights, and groundwork for predictive modelling.
Part 2 aims to train a model (Random Forest / XGBoost) to predict power from available features. 


## 🎯 Objectives

- Analyze the **seasonal and temporal variability** in wind power generation.
- Understand how **wind speed and direction** affect turbine output.
- Identify performance-critical **cut-in and rated wind speed thresholds**.
- Provide insights useful for **forecasting, optimization, and maintenance scheduling**.
- Set the foundation for **predictive modeling** and potential deployment.

## 📁 Project Structure

```bash
.
├── data/                  # Raw and cleaned datasets
├── notebooks/             # Jupyter notebooks for EDA
│   └── Wind-Turbine-Power-Generation-EDA.ipynb
├── models                 # Model outputs
├── scripts/               # Reusable Python functions
│   └── EDA_functions.py
├── figures/               # Generated plots and figures
├── environment.yml        # Conda environment file
├── README.md              # Project overview and documentation
└── .gitignore             # Git ignore rules

````

## 📊 Key Insights

![alt text](https://github.com/lasithagp/Wind-Turbine-Power-Generation-EDA/blob/main/figures/wind_turbine_analysis.png "wind_turbine_analysis")

![alt text](https://github.com/lasithagp/Wind-Turbine-Power-Generation-EDA/blob/main/figures/power_curves.png "power_curves")

* **Wind Speed vs Power**: Strong non-linear relationship observed; power increases sharply from 3–10 m/s and flattens after \~12 m/s.
* **Temporal Patterns**:

  * Power output is highest between **1:00–6:00**.
  * **Winter months** (Dec–Jan) show higher and more stable generation.
* **Wind Direction**:

  * Prevailing winds come from the **North-East quadrant**.
  * Directional changes influence power generation patterns.
* **Turbine Behavior**:

  * Power output drops to zero below ~ 3 m/s (cut-in) and saturates after ~12 m/s.
  * Possible misalignment or noise observed in outliers, useful for quality assurance.

## 📅 Features Engineered

* `wind_speed_diff`: difference between 100m and 10m wind speeds
* `hour`, `month`, `dayofweek`, `Season`: for diurnal and seasonal analysis
* `WD_bin`, `speed_bin`: for wind direction and speed analysis

---

## 📦 Tools & Libraries Used

* **Python 3.11**
* `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`
* `scikit-learn` (planned for modeling)
* `statsmodels` (for time decomposition)
* `scipy` (for correlation and statistical testing)
* `plotly` (3D visualization)

---

## Limitations

[Kaggle dataset](https://www.kaggle.com/datasets/sudhanvahg/wind-turbine-power-generation-forecasting/) limitations (e.g., no turbine specs, unknown location geography)

---

## 🧠 What's Next?

* 🔮 **Part-2: Modeling Stage** (in progress):

  * Train a regression model (Random Forest / XGBoost) to predict power from wind conditions and other available features in the dataset.
  * Evaluate model performance (MAE, R²).
  * Analyze feature importances using SHAP values.

* 🌐 **Deployment Stage**:

  * Build an interactive dashboard using Streamlit.
  * Let users explore power curves and seasonal trends dynamically.

## 🚀 Getting Started

1. Clone the repository:

```bash
git clone https://github.com/lasithagp/Wind-Turbine-Power-Generations-EDA.git
cd wind-power-eda
```

2. Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate windpower
```

3. Launch the notebook:

```bash
jupyter notebook notebooks/Wind-Turbine-Power-Generation-EDA.ipynb
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ About Me

I'm a Master's graduate in Data Science with a strong foundation in scientific analysis and environmental modeling. 
This project is part of my professional portfolio. 
I'm passionate about applying data science to real-world energy, climate, and sustainability challenges.

📫 Contact: \[[lasithaperera@yahoo.com](mailto:lasithaperera@yahoo.com), [lasithagp@gmail.com](mailto:lasithagp@gmail.com)] <br>
🔗 LinkedIn: \[[https://linkedin.com/in/lasitha-gonaduwage
](https://linkedin.com/in/lasitha-gonaduwage
)] <br>
Portfolio: \[[lasithagp.github.io/portfolio](https://lasithagp.github.io/portfolio/)]

