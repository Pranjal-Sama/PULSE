# ❤️ PULSE: Predictive Unified Learning System for Health Evaluation

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?logo=streamlit)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Ensemble%20Methods-green)
![Database](https://img.shields.io/badge/Database-SQLite-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-success)

**PULSE** is an advanced, production-grade machine learning web application designed to democratize preventive health screening. Developed as a Final Year B.Tech Project, it goes beyond simple binary predictions by integrating a **Multi-Disease Framework**, **Explainable AI (XAI)**, and a **Prescriptive Recommendation Engine** to provide actionable, risk-aware intelligence.

---

## 🌟 Key Features

* **Multi-Disease Diagnostic Framework:** Evaluates physiological data to predict the risk of Cardiovascular Disease, Type 2 Diabetes, and Chronic Kidney Disease (CKD).
* **Advanced Predictive Engine:** Utilizes highly tuned Ensemble models (Random Forest, XGBoost), heavily optimized for **Recall** to ensure minimal false negatives (missing an at-risk patient).
* **Explainable AI (XAI):** Integrates **SHAP (SHapley Additive exPlanations)** to transparently show users exactly *why* the model made its prediction, highlighting the specific vitals driving the risk score up or down.
* **Dynamic Actionable Intelligence:** Categorizes risk (Low 🟢, Medium 🟡, High 🔴) and prescribes dynamic, patient-specific advice spanning diet, exercise, and medical interventions.
* **Longitudinal Patient Tracking:** Features a built-in SQLite database to securely log and track patient risk trajectories across multiple visits over time.
* **Enterprise UI/UX:** A clean, responsive Streamlit dashboard featuring smart clinical alerts, a parameter reference guide, and one-click FPDF clinical report downloads.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit, HTML/CSS (Custom Theming)
* **Backend Data Processing:** Pandas, NumPy, SQLite3
* **Machine Learning:** Scikit-Learn (Random Forest, GridSearchCV, StandardScaler)
* **Explainability & Reporting:** SHAP, FPDF

---

## 📂 Repository Structure

```text
📦 PULSE
 ┣ 📂 models/                             # Serialized ML assets (*.pkl files for all 3 diseases)
 ┣ 📜 app.py                              # Main Streamlit web application & UI framework
 ┣ 📜 pulse_heart_model.py                # Heart Disease ML Training Pipeline
 ┣ 📜 pulse_diabetes_engine.py            # Diabetes ML Training Pipeline
 ┣ 📜 pulse_ckd_engine.py                 # Chronic Kidney Disease ML Pipeline
 ┣ 📜 Heart_disease_cleveland_new.csv     # Heart Disease clinical dataset
 ┣ 📜 diabetes.csv                        # Diabetes metabolic dataset
 ┣ 📜 Chronic_Kidney_Dsease_data.csv      # CKD renal dataset
 ┣ 📜 requirements.txt                    # Python dependencies
 ┗ 📜 README.md                           # Project documentation
🚀 How to Run Locally
Follow these steps to run the PULSE dashboard on your local machine:

1. Clone the repository

Bash
git clone [https://github.com/Pranjal-Sama/PULSE.git](https://github.com/Pranjal-Sama/PULSE.git)
cd PULSE
2. Create a virtual environment (Recommended)

Bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
3. Install dependencies
Note: We explicitly enforce numpy<2.0 to ensure compatibility with older compiled libraries.

Bash
pip install -r requirements.txt
4. Run the Streamlit Application

Bash
streamlit run app.py
The application will automatically open in your default web browser at http://localhost:8501.

🧠 The Machine Learning Pipelines
If you wish to re-train the models or explore the data analysis, the respective pulse_*_engine.py scripts contain the full pipelines for each disease:

Data Cleaning & Imputation: Handles skewed distributions (Median) and normal distributions (Mean) to preserve biological accuracy.

Exploratory Data Analysis (EDA): Generates distribution, correlation, and box plots.

Feature Engineering: Creates nonlinear clinical buckets (e.g., age_group, high_bp, GFR_Stage).

Model Selection: Uses GridSearchCV across Logistic Regression, Gradient Boosting, XGBoost, and Random Forest.

👨‍💻 Development Team
This system was developed as a Final Year B.Tech Project at the Department of Electrical & Computer Engineering, Bharati Vidyapeeth (Deemed to be University) College of Engineering, Pune.

Pranjal Sharma (PRN: 2214110546)

Vibhor Anshuman Roy (PRN: 2214110558)

Nagisetti Surya (PRN: 2214110556)

⚠️ Disclaimer: PULSE is an informational screening tool designed for early-stage risk assessment. It utilizes machine learning on physiological parameters. It is NOT a diagnostic medical device and should not replace professional medical consultation.
