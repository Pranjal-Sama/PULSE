import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# ==========================================
# 🟢 1. PAGE CONFIG & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="PULSE | Cardiovascular Risk",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        height: 3.2em; 
        background: linear-gradient(90deg, #ff4b4b, #ff6b6b); 
        color: white; 
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(255,75,75,0.3);
        border: none;
    }
    .stButton>button:hover { background: linear-gradient(90deg, #ff6b6b, #ff4b4b); }
    .risk-low { color: #00CC66; font-size: 28px; font-weight: 800; text-align: center; }
    .risk-med { color: #FFB300; font-size: 28px; font-weight: 800; text-align: center; }
    .risk-high { color: #FF3333; font-size: 28px; font-weight: 800; text-align: center; }
    .pulse-header { animation: pulse 2s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; } }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🟢 2. INITIALIZE SESSION STATE (2 Samples)
# ==========================================
default_values = {
    'age': 50, 'sex': 1, 'trestbps': 120, 'chol': 200, 'fbs': 0,
    'thalach': 150, 'cp': 1, 'exang': 0, 'oldpeak': 1.0, 
    'restecg': 1, 'slope': 1, 'ca': 0, 'thal': 1
}
for key, val in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = val

def load_lowest_risk():
    # Guaranteed low-risk: Young, athletic, female, optimal vitals, non-cardiac pain (cp=2 avoids Silent Ischemia flag)
    st.session_state.update({'age': 35, 'sex': 0, 'trestbps': 110, 'chol': 150, 'fbs': 0, 
                             'thalach': 180, 'cp': 2, 'exang': 0, 'oldpeak': 0.0, 
                             'restecg': 0, 'slope': 0, 'ca': 0, 'thal': 1})

def load_highest_risk():
    # Guaranteed high-risk: Older, male, severe clinical symptoms
    st.session_state.update({'age': 67, 'sex': 1, 'trestbps': 160, 'chol': 310, 'fbs': 1, 
                             'thalach': 108, 'cp': 1, 'exang': 1, 'oldpeak': 2.6, 
                             'restecg': 2, 'slope': 1, 'ca': 3, 'thal': 3})

# ==========================================
# 🟢 3. LOAD MODEL
# ==========================================
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open('models/heart_model.pkl', 'rb'))
        scaler = pickle.load(open('models/heart_scaler.pkl', 'rb'))
        features = pickle.load(open('models/heart_features.pkl', 'rb'))
        model_name = pickle.load(open('models/heart_model_name.pkl', 'rb'))
        return model, scaler, features, model_name
    except FileNotFoundError:
        st.error("❌ Model files not found. Please run the training engine first.")
        st.stop()

model, scaler, expected_features, model_name = load_assets()

# ==========================================
# 🟢 4. DYNAMIC & DETAILED RECOMMENDATION
# ==========================================
def _build_recommendation(risk_pct: float, inputs: dict):
    # Dynamic Triggers based on user input
    high_bp = inputs.get('trestbps', 120) > 130
    high_chol = inputs.get('chol', 200) > 240
    low_hr = inputs.get('thalach', 150) < 120

    if risk_pct <= 15:
        label = "LOW RISK 🟢"
        summary = "Your heart health indicators look good. Focus on maintaining your current healthy lifestyle."
        detail = {
            "Diet": [
                "Follow a heart-healthy diet: plenty of fruits, vegetables, and whole grains.",
                "Include omega-3 rich foods: salmon, walnuts, flaxseeds (2–3 times/week).",
                "Limit processed foods, trans fats, and excess sodium (< 2,300 mg/day).",
                "Aim for 8–10 glasses of water daily."
            ],
            "Exercise": [
                "Maintain 150 minutes of moderate aerobic activity per week.",
                "Surya Namaskar (Sun Salutation) — 12 rounds daily, morning.",
                "Pranayama (deep breathing): Anulom-Vilom 10 min/day — improves heart rate variability.",
                "Include strength training 2×/week to maintain muscle mass."
            ],
            "Lifestyle": [
                "Keep BMI within 18.5–24.9.",
                "Maintain blood pressure below 120/80 mmHg.",
                "Avoid smoking; limit alcohol to ≤ 1 drink/day.",
                "Sleep 7–9 hours nightly; poor sleep raises cardiovascular risk by 20%.",
                "Annual health check-up including lipid profile and ECG."
            ],
            "Natural Supplements (consult doctor first)": [
                "Arjuna bark (Terminalia arjuna) — traditional cardiac tonic.",
                "Garlic extract (600 mg/day) — helps maintain healthy cholesterol.",
                "Coenzyme Q10 (100 mg/day) — supports mitochondrial energy in heart cells."
            ]
        }
    elif risk_pct <= 40:
        label = "MEDIUM RISK 🟡"
        summary = "Your risk is moderate. Lifestyle modifications NOW can significantly reduce your chances of developing heart disease."
        detail = {
            "Diet (Strict)": [
                "DASH Diet or Mediterranean Diet — clinically proven to reduce CVD risk.",
                "Reduce sodium to < 1,500 mg/day if blood pressure is elevated.",
                "Eliminate trans fats completely; replace saturated fat with olive oil.",
                "Increase soluble fiber: oats, beans, lentils (aim for 25–30 g/day).",
                "Reduce refined sugar and simple carbohydrates — they elevate triglycerides.",
                "Add turmeric (curcumin) and ginger to daily cooking — anti-inflammatory."
            ],
            "Exercise (Structured)": [
                "Cardiac rehab-style routine: 30 min brisk walking × 5 days/week.",
                "Yoga sequence: Bhujangasana, Pawanmuktasana, Setu Bandhasana — 20 min/day.",
                "Avoid sudden intense exercise; build up gradually.",
                "Monitor heart rate during exercise — stay at 50–70% of max HR.",
                "Swimming and cycling — excellent low-impact cardio options."
            ],
            "Monitoring": [
                "Check blood pressure daily (home BP monitor recommended).",
                "Fasting lipid profile every 6 months.",
                "Blood glucose every 3–6 months (diabetes strongly linked to CVD).",
                "Visit a cardiologist for a baseline stress test / ECG."
            ],
            "Stress Management": [
                "Chronic stress raises cortisol → raises BP and inflammation.",
                "Meditation: 10–20 min daily (apps: Headspace, Calm).",
                "Limit work hours; take regular breaks."
            ],
            "Natural Supplements (consult doctor)": [
                "Ashwagandha (300–600 mg/day) — reduces cortisol and blood pressure.",
                "Omega-3 fish oil (2–4 g/day EPA+DHA) — lowers triglycerides significantly.",
                "Magnesium glycinate (300–400 mg) — supports cardiac muscle function.",
                "Hawthorn extract — traditional cardiac tonic, improves coronary blood flow.",
                "Red yeast rice — natural statin-like effect (consult physician first)."
            ]
        }
    else:
        label = "HIGH RISK 🔴"
        summary = "⚠️ URGENT: Your physiological indicators suggest HIGH risk of heart disease. Please consult a cardiologist IMMEDIATELY."
        detail = {
            "IMMEDIATE ACTIONS": [
                "Book an appointment with a cardiologist within the next 7 days.",
                "Request: ECG, 2D Echocardiogram, Stress Test, Full Lipid Panel, hsCRP.",
                "If you experience chest pain, breathlessness, or palpitations — call emergency services.",
                "Do NOT start new strenuous exercise without medical clearance."
            ],
            "Medical Management": [
                "Discuss with your doctor: statins (if high LDL), antihypertensives (if high BP),",
                "beta-blockers or ACE inhibitors depending on your specific profile.",
                "Ask about aspirin therapy — only under physician guidance.",
                "Monitor: BP twice daily, weight daily, symptoms diary."
            ],
            "Diet (Therapeutic)": [
                "Strict low-sodium diet: < 1,200 mg/day.",
                "Eliminate: fried foods, processed meats, full-fat dairy, alcohol.",
                "Eat: oily fish, legumes, leafy greens, berries, nuts in moderation.",
                "Small, frequent meals — avoid large meals that stress the heart.",
                "Maintain a food diary and share with your dietitian."
            ],
            "Gentle Movement Only": [
                "Only after medical clearance: gentle 15–20 min flat walks daily.",
                "NO heavy lifting, NO high-intensity workouts until cleared.",
                "Restorative Yoga only: Shavasana, gentle breathing exercises."
            ],
            "Emotional & Mental Health": [
                "High-risk diagnosis is stressful — consider counselling.",
                "Involve family/support network in your care plan.",
                "Depression and anxiety worsen cardiac outcomes — address them proactively."
            ],
            "Natural Supplements (ALONGSIDE medical treatment)": [
                "Omega-3 (under doctor supervision — may interact with blood thinners).",
                "Coenzyme Q10 (especially if on statin — statins deplete CoQ10).",
                "NEVER stop prescribed medication in favor of supplements."
            ]
        }

    # Inject dynamic user-specific warnings
    if high_bp:
        detail.setdefault("⚠️ Specific Warning", []).append(f"Your Resting BP ({inputs['trestbps']} mmHg) is dangerously elevated. Prioritize sodium reduction.")
    if high_chol:
        detail.setdefault("⚠️ Specific Warning", []).append(f"Your Serum Cholesterol ({inputs['chol']} mg/dL) requires immediate dietary intervention to reduce plaque buildup.")
    if low_hr:
        detail.setdefault("⚠️ Specific Warning", []).append(f"Your Max Heart Rate achieved ({inputs['thalach']} bpm) is notably low, indicating poor cardiovascular endurance.")

    return {'risk_pct': round(risk_pct, 2), 'risk_label': label, 'advice_summary': summary, 'advice_detail': detail}

# ==========================================
# 🟢 5. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=90)
    st.title("PULSE System")
    st.caption("Predictive Unified Learning System for Health Evaluation")
    st.markdown("---")
    page = st.radio("Navigation", ["🩺 Risk Assessment", "📖 Clinical Reference Guide", "ℹ️ About PULSE"])
    st.markdown("---")
    st.warning("⚠️ **Screening Tool Only.**\nPULSE is NOT a diagnostic device. Always consult a physician.")

# ==========================================
# 🟢 PAGE 1: RISK ASSESSMENT
# ==========================================
if page == "🩺 Risk Assessment":
    st.markdown("<h1 class='pulse-header' style='text-align:center; color:#ff4b4b;'>❤️ Cardiovascular Risk Assessment</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'><b>PULSE</b> — Early detection of heart disease risk using machine learning<br>(B.Tech Final Year Project, Bharati Vidyapeeth COE Pune)</p>", unsafe_allow_html=True)
    st.write("---")

    st.write("**Quick Demo:** Select a profile to load predefined data.")
    colA, colB = st.columns(2)
    with colA:
        st.button("🟢 Load Healthy Patient (Low Risk)", on_click=load_lowest_risk)
    with colB:
        st.button("🔴 Load Symptomatic Patient (High Risk)", on_click=load_highest_risk)

    with st.form("pulse_form"):
        st.subheader("Patient Physiological Parameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", 20, 100, key='age')
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female", key='sex')
            trestbps = st.number_input("Resting BP (mmHg)", 80, 220, key='trestbps')
        with c2:
            chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, key='chol')
            fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='fbs')
            thalach = st.number_input("Max Heart Rate Achieved", 60, 220, key='thalach')
        with c3:
            # Updated tooltip to include Silent Ischemia warning
            cp = st.selectbox("Chest Pain Type", [0,1,2,3], key='cp', help="0=Typical Angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic (⚠️ Silent Ischemia Risk)")
            exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='exang')
            oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, step=0.1, key='oldpeak')

        st.markdown("---")
        st.subheader("Advanced Cardiac Indicators")
        c4, c5, c6 = st.columns(3)
        with c4:
            restecg = st.selectbox("Resting ECG", [0,1,2], key='restecg')
            slope = st.selectbox("Slope of ST Segment", [0,1,2], key='slope')
        with c5:
            ca = st.selectbox("Major Vessels (Fluoroscopy)", [0,1,2,3], key='ca')
            thal = st.selectbox("Thalassemia", [1,2,3], key='thal')
        with c6:
            st.write("")
            st.write("")
            submitted = st.form_submit_button("🚀 Analyze with PULSE Engine")

    if submitted:
        with st.spinner("Running PULSE prediction engine..."):
            raw_inputs = {
                'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
                'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
                'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
            }
            df_input = pd.DataFrame([raw_inputs])

            # EXACT feature engineering matched to your training script
            df_input['age_group'] = pd.cut(df_input['age'], bins=[0, 40, 50, 60, 70, 100], labels=[0, 1, 2, 3, 4]).astype(int)
            df_input['high_bp'] = (df_input['trestbps'] > 130).astype(int)
            df_input['high_chol'] = (df_input['chol'] > 240).astype(int)
            df_input['low_thalach'] = (df_input['thalach'] < 120).astype(int)
            df_input['oldpeak_cat'] = pd.cut(df_input['oldpeak'], bins=[-0.1, 0, 1, 2, 10], labels=[0, 1, 2, 3]).astype(int)

            for col in expected_features:
                if col not in df_input.columns: df_input[col] = 0
            X_live = df_input[expected_features]
            X_scaled = scaler.transform(X_live)
            
            prob = model.predict_proba(X_scaled)[0][1]
            result = _build_recommendation(prob * 100, raw_inputs)

        # Display Results
        st.markdown("---")
        st.header("📋 PULSE Diagnostic Report")
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.metric("Calculated Risk Probability", f"{result['risk_pct']:.1f}%")
            if result['risk_pct'] <= 15: 
                st.markdown(f"<div class='risk-low'>{result['risk_label']}</div>", unsafe_allow_html=True)
            elif result['risk_pct'] <= 40: 
                st.markdown(f"<div class='risk-med'>{result['risk_label']}</div>", unsafe_allow_html=True)
            else: 
                st.markdown(f"<div class='risk-high'>{result['risk_label']}</div>", unsafe_allow_html=True)
            st.progress(min(result['risk_pct'] / 100, 1.0))

        with col_right:
            st.subheader("Actionable Intelligence")
            st.info(result['advice_summary'])
            for category, tips in result['advice_detail'].items():
                with st.expander(f"**{category}**", expanded=True):
                    for tip in tips:
                        st.write(f"• {tip}")

        report_text = f"PULSE Report\nGenerated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}\nRisk: {result['risk_pct']:.1f}%\nSummary: {result['advice_summary']}"
        st.download_button("📄 Download PDF/TXT Report", report_text, file_name=f"PULSE_Report_{datetime.now().strftime('%Y%m%d')}.txt")

# ==========================================
# 🟢 PAGE 2: CLINICAL REFERENCE GUIDE
# ==========================================
elif page == "📖 Clinical Reference Guide":
    st.title("📖 Clinical Reference Guide")
    st.markdown("Understanding the input parameters is critical for accurate risk assessment. Below is a dictionary of the clinical terms used in the PULSE system.")
    
    st.header("The Basics")
    st.markdown("""
    * **Resting Blood Pressure (trestbps):** Measured in mmHg. Normal is <120. Hypertension Stage 1 is 130-139, and Stage 2 is 140 or higher.
    * **Serum Cholesterol (chol):** Measured in mg/dL. Optimal is under 200. Readings over 240 indicate high cardiovascular risk.
    * **Fasting Blood Sugar (fbs):** A reading > 120 mg/dL is an indicator of potential diabetes.
    """)
    
    st.header("Complex Diagnostics")
    st.markdown("""
    * **Chest Pain Type (cp):**
        * `0` **Typical Angina:** Classic chest pain caused by reduced blood flow to the heart.
        * `1` **Atypical Angina:** Chest pain not fitting classic criteria.
        * `2` **Non-anginal Pain:** Pain not related to the heart (e.g., muscle spasms or acid reflux).
        * `3` **Asymptomatic (⚠️ High Risk Indicator):** No chest pain present. *Note: In clinical screening datasets like the one PULSE uses, an asymptomatic patient who still requires a cardiac workup often suffers from **Silent Ischemia**—a severe, painless blockage that carries a highly elevated risk of sudden cardiac events.*
    * **Resting ECG (restecg):**
        * `0` **Normal:** Healthy electrical activity.
        * `1` **ST-T Wave Abnormality:** Minor arrhythmias or irregular electrical signals.
        * `2` **Left Ventricular Hypertrophy:** Enlargement of the heart's main pumping chamber.
    * **Thalassemia (thal):**
        * `1` **Normal:** Healthy blood flow.
        * `2` **Fixed Defect:** A past heart attack has caused permanent tissue damage.
        * `3` **Reversible Defect:** A current issue where blood flow is restricted during exercise.
    * **ST Depression & Slope:**
        * **Oldpeak:** Measures how far the ST segment on an ECG drops during exercise. A higher number indicates severe ischemia (lack of oxygen).
        * **Slope:** Looks at the trajectory of the ECG wave. `0` (Upsloping) is healthy. `1` (Flat) or `2` (Downsloping) are strong indicators of an unhealthy heart.
    """)

# ==========================================
# 🟢 PAGE 3: ABOUT PULSE
# ==========================================
elif page == "ℹ️ About PULSE":
    st.title("ℹ️ About the PULSE Project")
    st.markdown("""
    **PULSE (Predictive Unified Learning System for Health Evaluation)** is an advanced machine learning framework designed to democratize preventive health screening. 
    Rather than simply predicting binary outcomes, PULSE integrates rigorous algorithmic evaluation with a prescriptive recommendation engine to provide actionable, risk-aware intelligence.
    
    ### 👨‍💻 Development Team
    This system was developed as a Final Year B.Tech Project at the Department of Electrical & Computer Engineering.
    * **Pranjal Sharma** (PRN: 2214110546)
    * **Vibhor Anshuman Roy** (PRN: 2214110558)
    * **Nagisetti Surya** (PRN: 2214110556)
    
    ### 🎓 Mentorship
    * **Project Guide:** Prof. Dr. Datta Chavan 
    * **Institution:** Bharati Vidyapeeth (Deemed to be University) College of Engineering, Pune.
    """)

st.markdown("---")
st.markdown("<p style='text-align:center; color:gray; font-size:small;'><b>PULSE</b> — B.Tech Final Year Project | Bharati Vidyapeeth COE Pune</p>", unsafe_allow_html=True)