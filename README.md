# ❤️ PULSE: Predictive Unified Learning System for Health Evaluation

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?logo=streamlit)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Active-success)

**PULSE** is an advanced, production-grade machine learning web application designed to democratize preventive cardiovascular health screening. Developed as a Final Year B.Tech Project, it goes beyond simple binary predictions by integrating **Explainable AI (XAI)** and a **Prescriptive Recommendation Engine** to provide actionable, risk-aware intelligence.

---

## 🌟 Key Features

* **Advanced Predictive Engine:** Utilizes a highly tuned **Random Forest Classifier**, heavily optimized for **Recall (92.86%)** to ensure minimal false negatives (missing a sick patient).
* **Dynamic Actionable Intelligence:** Does not just output a probability. It categorizes risk (Low 🟢, Medium 🟡, High 🔴) and prescribes dynamic, patient-specific advice spanning Diet, Exercise, and Medical interventions.
* **Explainable AI (XAI):** Integrates **SHAP (SHapley Additive exPlanations)** and Plotly visualization to transparently show users exactly *why* the model made its prediction, highlighting the specific vitals driving the risk score up or down.
* **Clinical Data Awareness:** Correctly identifies and adjusts for complex clinical phenomena within the Cleveland dataset, such as **Silent Ischemia** (where asymptomatic patients face severe risk).
* **Enterprise UI/UX:** A clean, responsive Streamlit dashboard featuring smart tooltips, a clinical reference guide, and one-click PDF report downloads.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit, HTML/CSS (Custom Theming)
* **Backend Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest, GridSearchCV, StandardScaler)
* **Explainability & Visualization:** SHAP, Plotly, Matplotlib, Seaborn

---

## 📂 Repository Structure

```text
📦 PULSE-Cardio-Risk
 ┣ 📂 models/                             # Serialized ML assets
 ┃ ┣ 📜 heart_model.pkl                   # Trained Random Forest model
 ┃ ┣ 📜 heart_scaler.pkl                  # Fitted StandardScaler
 ┃ ┣ 📜 heart_features.pkl                # Ordered list of 16 engineered features
 ┃ ┗ 📜 heart_model_name.pkl              # Model metadata
 ┣ 📜 app.py                              # Main Streamlit web application
 ┣ 📜 pulse_heart_model.py                # ML Training Pipeline & EDA script
 ┣ 📜 Heart_disease_cleveland_new.csv     # Raw clinical dataset
 ┣ 📜 requirements.txt                    # Python dependencies
 ┗ 📜 README.md                           # Project documentation

## 🚀 How to Run Locally

Follow these steps to run the PULSE dashboard on your local machine:

**1. Clone the repository**
```bash
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

🧠 The Machine Learning Pipeline (pulse_heart_model.py)
If you wish to re-train the model or explore the data analysis, the pulse_heart_model.py script contains the full pipeline:

Data Cleaning & Imputation: Handles skewed distributions and missing values.

Exploratory Data Analysis (EDA): Generates distribution, correlation, and box plots.

Feature Engineering: Creates nonlinear buckets (e.g., age_group, high_bp, oldpeak_cat).

Model Selection: Uses GridSearchCV across Logistic Regression, Gradient Boosting, XGBoost, and Random Forest. F1-Score and Recall were the primary selection metrics.

👨‍💻 Development Team
This system was developed as a Final Year B.Tech Project at the Department of Electrical & Computer Engineering, Bharati Vidyapeeth (Deemed to be University) College of Engineering, Pune.

Pranjal Sharma (PRN: 2214110546)

Vibhor Anshuman Roy (PRN: 2214110558)

Nagisetti Surya (PRN: 2214110556)

Mentorship:

Project Guide: Prof. Dr. Datta Chavan

⚠️ Disclaimer: PULSE is an informational screening tool designed for early-stage risk assessment. It utilizes machine learning on physiological parameters. It is NOT a diagnostic medical device and should not replace professional medical consultation.
