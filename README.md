# 🎯 Target Marketing: EDA and Classification

This project analyzes a marketing dataset containing customer phone call records used to promote a deposit product. The aim was to understand factors influencing successful conversions and build baseline classification models to predict them.

---

## 📊 Project Overview

- Conducted **Exploratory Data Analysis (EDA)** to uncover trends in customer demographics, call outcomes, and campaign strategy.
- Modeled the conversion outcome (binary classification: deposit or not) using:
  - **Logistic Regression**
  - **Decision Tree Classifier**
- Applied **cross-validation** to evaluate performance and reduce model variance.

---

## 🔍 Key Findings from EDA

- Conversion rate was low, confirming **class imbalance** in the data.
- Features like `contact_type`, `previous_outcome`, `duration`, and `month` were strong indicators of success.
- Call success rates varied significantly by age group, marital status, and previous interactions.

---

## ⚙️ Tools and Technologies

- Python: `pandas`, `numpy`, `seaborn`, `matplotlib`
- Modeling: `scikit-learn`
- Jupyter Notebooks for analysis and modeling

## 📄 License
This project is licensed under the MIT License.
