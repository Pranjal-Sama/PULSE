import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import sqlite3
import os
from datetime import datetime
from fpdf import FPDF

# --- New Imports for Doctor DB & Email Integration ---
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- New Imports for Client-Side Geolocation ---
import streamlit.components.v1 as components
from components.location import get_nearest_city, reverse_geocode_coordinates

# ==========================================
# 🟢 1. PAGE CONFIG & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="PULSE | Health Evaluation",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* ===== GLOBAL STYLING ===== */
    * {
        margin: 0;
        padding: 0;
    }
    
    body {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #e2e8f0;
    }
    
    /* ===== SIDEBAR STYLING ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1a2744 100%);
        padding: 30px 20px;
    }
    
    [data-testid="stSidebar"] [data-testid="stImage"] {
        border-radius: 50%;
        padding: 10px;
        background: rgba(0, 212, 255, 0.15);
        border: 2px solid rgba(0, 212, 255, 0.3);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        color: #e0f2fe !important;
    }
    
    /* ===== BUTTON STYLING ===== */
    .stButton>button { 
        width: 100%; 
        border-radius: 12px; 
        height: 3.2em; 
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: #0f172a; 
        font-weight: 700;
        font-size: 16px;
        box-shadow: 0 8px 20px rgba(0, 212, 255, 0.4);
        border: 2px solid #00d4ff;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .stButton>button:hover { 
        background: linear-gradient(135deg, #00ffff 0%, #00ccff 100%);
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.6);
        transform: translateY(-3px);
        color: #0f172a;
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    /* ===== RISK LEVEL STYLING ===== */
    .risk-low { 
        color: #4ade80; 
        font-size: 32px; 
        font-weight: 900; 
        text-align: center;
        text-shadow: 0 0 20px rgba(74, 222, 128, 0.4);
        animation: pulse-glow 2s ease-in-out infinite;
    }
    
    .risk-med { 
        color: #fbbf24; 
        font-size: 32px; 
        font-weight: 900; 
        text-align: center;
        text-shadow: 0 0 20px rgba(251, 191, 36, 0.4);
        animation: pulse-glow 2s ease-in-out infinite;
    }
    
    .risk-high { 
        color: #ff6b6b; 
        font-size: 32px; 
        font-weight: 900; 
        text-align: center;
        text-shadow: 0 0 20px rgba(255, 107, 107, 0.5);
        animation: pulse-glow 2s ease-in-out infinite;
    }
    
    @keyframes pulse-glow {
        0%, 100% { 
            transform: scale(1);
        }
        50% { 
            transform: scale(1.05);
        }
    }
    
    /* ===== ALERT BOX STYLING ===== */
    .alert-box { 
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        padding: 18px 20px; 
        border-left: 6px solid #ff6b6b;
        border-radius: 10px; 
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(255, 107, 107, 0.25);
        font-weight: 600;
        color: #fecaca;
        border: 1px solid rgba(255, 107, 107, 0.4);
    }
    
    /* ===== FORM STYLING ===== */
    .stForm {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.15);
        border: 2px solid rgba(0, 212, 255, 0.3);
    }
    
    /* ===== INPUT STYLING ===== */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        background-color: #0f172a !important;
        color: #e2e8f0 !important;
        border: 2px solid #00d4ff !important;
        border-radius: 10px !important;
        padding: 12px !important;
        font-size: 15px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus {
        border-color: #00ffff !important;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.4) !important;
        background-color: #1a2744 !important;
        color: #e2e8f0 !important;
    }
    
    /* ===== METRIC CARDS ===== */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid #00d4ff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.2);
    }
    
    /* ===== EXPANDER STYLING ===== */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #0099cc 0%, #00d4ff 100%);
        color: #0f172a !important;
        border-radius: 10px;
        font-weight: 700;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    .streamlit-expanderContent {
        background: linear-gradient(135deg, #1a2744 0%, #0f172a 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        padding: 15px;
        color: #e2e8f0;
    }
    
    /* ===== HEADING STYLING ===== */
    h1, h2, h3, h4, h5, h6 {
        color: #e0f2fe !important;
        font-weight: 700 !important;
        margin-top: 20px !important;
        margin-bottom: 15px !important;
    }
    
    h1 {
        font-size: 2.5em !important;
        color: #00ffff !important;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
    }
    
    h2 {
        font-size: 1.8em !important;
        border-bottom: 3px solid #00d4ff;
        padding-bottom: 10px;
        color: #e0f2fe !important;
    }
    
    h3 {
        color: #00ffff !important;
    }
    
    h4 {
        color: #e0f2fe !important;
    }
    
    p {
        color: #cbd5e1 !important;
    }
    
    /* ===== INFO BOX STYLING ===== */
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 12px !important;
        padding: 18px !important;
        font-weight: 500 !important;
        border-left: 6px solid !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #0c4a6e 0%, #082f4f 100%) !important;
        border-left-color: #00d4ff !important;
        color: #e0f2fe !important;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #134e4a 0%, #0d3830 100%) !important;
        border-left-color: #4ade80 !important;
        color: #d1fae5 !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #78350f 0%, #451a03 100%) !important;
        border-left-color: #fbbf24 !important;
        color: #fef3c7 !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #7f1d1d 0%, #4c0519 100%) !important;
        border-left-color: #ff6b6b !important;
        color: #fecaca !important;
    }
    
    /* ===== PROGRESS BAR ===== */
    .stProgress>div>div>div {
        background: linear-gradient(90deg, #4ade80 0%, #00d4ff 100%) !important;
        border-radius: 10px !important;
    }
    
    /* ===== DIVIDER ===== */
    hr {
        border: 0;
        height: 2px;
        background: linear-gradient(90deg, #0099cc 0%, #00d4ff 50%, #0099cc 100%);
        margin: 30px 0;
    }
    
    /* ===== CUSTOM GRADIENT BACKGROUND ===== */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1a2744 100%);
    }
    
    /* ===== TOAST/SPINNER ===== */
    .stSpinner>div {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
    }
    
    /* ===== TEXT COLOR FIX ===== */
    .stMarkdown {
        color: #cbd5e1 !important;
    }
    
    ul li {
        color: #cbd5e1 !important;
    }
    
    /* ===== TABS STYLING ===== */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #0f172a; border-radius: 10px 10px 0 0; padding: 10px 20px; border: 1px solid #00d4ff; border-bottom: none; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #0099cc 0%, #00d4ff 100%); color: #0f172a !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🟢 2. DATABASE INITIALIZATION (SQLite)
# ==========================================
def init_db():
    conn = sqlite3.connect('pulse_patients.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patient_history 
                 (timestamp TEXT, patient_id TEXT, disease_module TEXT, risk_score REAL)''')
    conn.commit()
    conn.close()

init_db()

def save_patient_record(patient_id, module, risk_score):
    conn = sqlite3.connect('pulse_patients.db')
    c = conn.cursor()
    c.execute("INSERT INTO patient_history VALUES (?, ?, ?, ?)", 
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), patient_id, module, risk_score))
    conn.commit()
    conn.close()

# ==========================================
# 🟢 3. PDF GENERATION ENGINE
# ==========================================
def create_pdf_report(patient_id, disease, risk_pct, risk_label, summary, details):
    def clean_text(text):
        return str(text).encode('latin-1', 'ignore').decode('latin-1').strip()
        
    safe_disease = clean_text(disease)
    safe_risk_label = clean_text(risk_label)
    safe_summary = clean_text(summary)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="PULSE Clinical Diagnostic Report", ln=True, align='C')
    
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(100, 10, txt=f"Patient ID: {clean_text(patient_id)}", ln=False)
    pdf.cell(100, 10, txt=f"Module: {safe_disease}", ln=True)
    pdf.cell(100, 10, txt=f"Calculated Risk: {risk_pct}%", ln=False)
    pdf.cell(100, 10, txt=f"Risk Tier: {safe_risk_label}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Clinical Summary:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8, txt=safe_summary)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Actionable Intelligence:", ln=True)
    pdf.set_font("Arial", '', 11)
    
    for category, tips in details.items():
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(200, 8, txt=f"{clean_text(category)}:", ln=True)
        pdf.set_font("Arial", '', 11)
        for tip in tips:
            pdf.multi_cell(0, 8, txt=f"  - {clean_text(tip)}")
    
    filename = f"temp_report_{clean_text(patient_id)}.pdf"
    pdf.output(filename)
    with open(filename, "rb") as f:
        bytes_data = f.read()
    os.remove(filename)
    return bytes_data

# ==========================================
# 🟢 4. CLINICAL ALERTS ENGINE
# ==========================================
def check_clinical_alerts(disease, inputs):
    alerts = []
    if disease == "❤️ Heart Disease":
        if inputs.get('trestbps', 0) > 160 and inputs.get('chol', 0) > 280:
            alerts.append("CRITICAL: Concurrent Stage 2 Hypertension and Severe Hypercholesterolemia detected. Immediate cardiovascular workup required.")
    elif disease == "🩸 Diabetes":
        if inputs.get('Glucose', 0) > 200:
            alerts.append("CRITICAL: Fasting Plasma Glucose > 200 mg/dL. Severe hyperglycemia detected. Risk of diabetic ketoacidosis.")
    elif disease == "🫘 Chronic Kidney Disease":
        if inputs.get('GFR', 100) < 30 and inputs.get('SerumCreatinine', 0) > 2.0:
            alerts.append("CRITICAL: GFR indicates Stage 4/5 Severe Renal Failure. Urgent nephrology consultation and dialysis evaluation required.")
    return alerts

# ==========================================
# 🟢 5. ASSET LOADING (CACHED)
# ==========================================
@st.cache_resource
def load_assets(prefix):
    try:
        model = pickle.load(open(f'models/{prefix}_model.pkl', 'rb'))
        scaler = pickle.load(open(f'models/{prefix}_scaler.pkl', 'rb'))
        features = pickle.load(open(f'models/{prefix}_features.pkl', 'rb'))
        return model, scaler, features
    except FileNotFoundError:
        return None, None, None

heart_model, heart_scaler, heart_features = load_assets('heart')
diab_model, diab_scaler, diab_features = load_assets('diabetes')
ckd_model, ckd_scaler, ckd_features = load_assets('ckd')

# ==========================================
# 🟢 6. RECOMMENDATION ENGINES (FULL DETAIL)
# ==========================================
def _build_heart_recommendation(risk_pct, inputs):
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

    if high_bp:
        detail.setdefault("⚠️ Specific Warning", []).append(f"Your Resting BP ({inputs['trestbps']} mmHg) is dangerously elevated. Prioritize sodium reduction.")
    if high_chol:
        detail.setdefault("⚠️ Specific Warning", []).append(f"Your Serum Cholesterol ({inputs['chol']} mg/dL) requires immediate dietary intervention to reduce plaque buildup.")
    if low_hr:
        detail.setdefault("⚠️ Specific Warning", []).append(f"Your Max Heart Rate achieved ({inputs['thalach']} bpm) is notably low, indicating poor cardiovascular endurance.")

    return {'risk_pct': round(risk_pct, 2), 'risk_label': label, 'advice_summary': summary, 'advice_detail': detail}

def _build_diabetes_recommendation(risk_pct, inputs):
    if risk_pct <= 25:
        label = "LOW RISK 🟢"
        summary = "Your glycemic and metabolic indicators are stable. Maintain your healthy habits."
        detail = {
            "Diet": ["Maintain a balanced diet rich in complex carbohydrates and fiber.", "Drink water primarily; avoid sugary sodas and juices."],
            "Lifestyle": ["Aim for 150 minutes of moderate exercise per week.", "Ensure 7-8 hours of sleep to maintain healthy metabolic hormones."]
        }
    elif risk_pct <= 50:
        label = "MEDIUM RISK 🟡 (Possible Prediabetes)"
        summary = "You show signs of metabolic stress. Action taken now can prevent progression to Type 2 Diabetes."
        detail = {
            "Medical Checks": ["Schedule a Fasting Blood Glucose and HbA1c test with your doctor."],
            "Dietary Overhaul": ["Switch completely to complex carbs.", "Increase fiber intake significantly."],
            "Exercise": ["Incorporate strength training 2-3 times a week.", "Take a 10-15 minute walk immediately after large meals to lower post-meal blood sugar."]
        }
    else:
        label = "HIGH RISK 🔴"
        summary = "⚠️ URGENT: Your indicators strongly suggest the presence of Diabetes. Please consult an endocrinologist immediately."
        detail = {
            "IMMEDIATE ACTIONS": ["Book an appointment with a doctor for a definitive HbA1c and Oral Glucose Tolerance Test (OGTT)."],
            "Diet (Strict)": ["Eliminate all simple sugars, sweets, and refined carbohydrates immediately.", "Adopt a strict low-glycemic or clinically managed low-carb diet."],
            "Complication Prevention": ["Check your feet daily for cuts or sores (diabetic neuropathy risk).", "Schedule a comprehensive eye exam (diabetic retinopathy risk)."]
        }
    return {'risk_pct': round(risk_pct, 2), 'risk_label': label, 'advice_summary': summary, 'advice_detail': detail}

def _build_ckd_recommendation(risk_pct, inputs):
    if risk_pct <= 20:
        label = "LOW RISK 🟢"
        summary = "Your kidney function indicators look normal. Maintain your current healthy lifestyle."
        detail = {
            "Hydration": ["Drink adequate water daily (approx 2-3 liters) to help kidneys clear sodium and toxins."],
            "Diet & Lifestyle": ["Limit the use of over-the-counter painkillers like Ibuprofen/NSAIDs, which can stress the kidneys over time."],
            "Monitoring": ["Continue with annual routine checkups including basic metabolic panels."]
        }
    elif risk_pct <= 50:
        label = "MEDIUM RISK 🟡 (Renal Stress)"
        summary = "You have markers indicating possible early-stage renal stress. Early intervention is key."
        detail = {
            "Medical Actions": ["Schedule a follow-up test for Serum Creatinine, BUN, and a urinalysis for protein (proteinuria).", "Ensure strict control of your blood pressure and blood sugar."],
            "Dietary Changes": ["Adopt a moderately low-sodium diet (< 2,300 mg/day) to reduce fluid retention and BP.", "Avoid crash diets or extremely high-protein diets."]
        }
    else:
        label = "HIGH RISK 🔴"
        summary = "⚠️ URGENT: Your physiological indicators strongly suggest compromised kidney function. Consult a Nephrologist immediately."
        detail = {
            "IMMEDIATE ACTIONS": ["Book an appointment with a Nephrologist within the next 7 days.", "Request a comprehensive Renal Panel, GFR assessment, and Renal Ultrasound."],
            "Strict Renal Diet": ["Low Sodium: Avoid all processed foods, canned soups, and fast food.", "Controlled Protein: Work with a renal dietitian to consume the exact right amount of protein."],
            "Symptom Monitoring": ["Watch for swelling in the ankles/legs (edema), extreme fatigue, metallic taste in the mouth, or changes in urination frequency."]
        }
    return {'risk_pct': round(risk_pct, 2), 'risk_label': label, 'advice_summary': summary, 'advice_detail': detail}

# ==========================================
# 🟢 7. DOCTOR DATABASE & ML RANKING ENGINE
# ==========================================
MOCK_DOCTORS = {
    "Mumbai": {
        "❤️ Heart Disease": [
            {"name": "Dr. Anil Shah", "exp_yrs": 25, "rating": 4.8, "fee": 2500, "dist_km": 4.2, "address": "Breach Candy, Mumbai", "time": "10 AM - 4 PM"},
            {"name": "Dr. Riya Mehta", "exp_yrs": 12, "rating": 4.6, "fee": 1500, "dist_km": 8.5, "address": "Andheri West, Mumbai", "time": "5 PM - 9 PM"},
            {"name": "Dr. Vikram Joshi", "exp_yrs": 30, "rating": 4.9, "fee": 3000, "dist_km": 12.0, "address": "Bandra, Mumbai", "time": "9 AM - 1 PM"}
        ],
        "🩸 Diabetes": [
            {"name": "Dr. Sneha Patil", "exp_yrs": 18, "rating": 4.7, "fee": 1200, "dist_km": 3.1, "address": "Dadar, Mumbai", "time": "11 AM - 3 PM"},
            {"name": "Dr. Karan Desai", "exp_yrs": 8, "rating": 4.4, "fee": 800, "dist_km": 15.0, "address": "Borivali, Mumbai", "time": "4 PM - 8 PM"},
            {"name": "Dr. Meena Iyer", "exp_yrs": 22, "rating": 4.8, "fee": 1800, "dist_km": 6.4, "address": "Juhu, Mumbai", "time": "10 AM - 2 PM"}
        ],
        "🫘 Chronic Kidney Disease": [
            {"name": "Dr. Rajiv Menon", "exp_yrs": 20, "rating": 4.7, "fee": 2200, "dist_km": 7.8, "address": "Powai, Mumbai", "time": "12 PM - 6 PM"},
            {"name": "Dr. Anjali Verma", "exp_yrs": 15, "rating": 4.5, "fee": 1600, "dist_km": 9.2, "address": "Goregaon, Mumbai", "time": "9 AM - 1 PM"}
        ]
    },
    "Kolkata": {
        "❤️ Heart Disease": [
            {"name": "Dr. Amitava Sen", "exp_yrs": 28, "rating": 4.9, "fee": 2000, "dist_km": 5.5, "address": "Salt Lake Sector V, Kolkata", "time": "10 AM - 2 PM"},
            {"name": "Dr. B. Biswas", "exp_yrs": 14, "rating": 4.5, "fee": 1000, "dist_km": 11.2, "address": "Park Street, Kolkata", "time": "4 PM - 8 PM"},
            {"name": "Dr. S. Mukherjee", "exp_yrs": 20, "rating": 4.7, "fee": 1500, "dist_km": 3.0, "address": "Gariahat, Kolkata", "time": "11 AM - 5 PM"}
        ],
        "🩸 Diabetes": [
            {"name": "Dr. Tanya Das", "exp_yrs": 16, "rating": 4.8, "fee": 1200, "dist_km": 8.1, "address": "Ballygunge, Kolkata", "time": "9 AM - 1 PM"},
            {"name": "Dr. Rahul Roy", "exp_yrs": 10, "rating": 4.3, "fee": 800, "dist_km": 14.5, "address": "Dum Dum, Kolkata", "time": "5 PM - 9 PM"}
        ],
        "🫘 Chronic Kidney Disease": [
            {"name": "Dr. Pratik Saha", "exp_yrs": 24, "rating": 4.8, "fee": 1800, "dist_km": 6.7, "address": "New Town, Kolkata", "time": "10 AM - 4 PM"},
            {"name": "Dr. M. Banerjee", "exp_yrs": 12, "rating": 4.6, "fee": 1100, "dist_km": 4.2, "address": "Howrah, Kolkata", "time": "3 PM - 7 PM"}
        ]
    },
    "Delhi": {
        "❤️ Heart Disease": [
            {"name": "Dr. R.K. Sharma", "exp_yrs": 32, "rating": 4.9, "fee": 3000, "dist_km": 9.5, "address": "Vasant Kunj, Delhi", "time": "10 AM - 4 PM"},
            {"name": "Dr. Nidhi Gupta", "exp_yrs": 15, "rating": 4.6, "fee": 1800, "dist_km": 4.1, "address": "Lajpat Nagar, Delhi", "time": "9 AM - 2 PM"},
            {"name": "Dr. Vikas Singh", "exp_yrs": 9, "rating": 4.2, "fee": 1000, "dist_km": 18.0, "address": "Rohini, Delhi", "time": "4 PM - 8 PM"}
        ],
        "🩸 Diabetes": [
            {"name": "Dr. Alok Verma", "exp_yrs": 21, "rating": 4.8, "fee": 1500, "dist_km": 7.2, "address": "Connaught Place, Delhi", "time": "11 AM - 5 PM"},
            {"name": "Dr. Sunita Rao", "exp_yrs": 14, "rating": 4.5, "fee": 1200, "dist_km": 12.3, "address": "Dwarka, Delhi", "time": "9 AM - 1 PM"},
            {"name": "Dr. Aman Khurana", "exp_yrs": 7, "rating": 4.1, "fee": 700, "dist_km": 5.0, "address": "Karol Bagh, Delhi", "time": "5 PM - 9 PM"}
        ],
        "🫘 Chronic Kidney Disease": [
            {"name": "Dr. D. Malhotra", "exp_yrs": 28, "rating": 4.9, "fee": 2500, "dist_km": 10.1, "address": "Saket, Delhi", "time": "10 AM - 3 PM"},
            {"name": "Dr. Pooja Chawla", "exp_yrs": 17, "rating": 4.7, "fee": 1600, "dist_km": 6.8, "address": "South Ex, Delhi", "time": "4 PM - 8 PM"}
        ]
    },
    "New York": {
        "❤️ Heart Disease": [
            {"name": "Dr. James Wilson", "exp_yrs": 26, "rating": 4.9, "fee": 400, "dist_km": 3.2, "address": "Manhattan, NY", "time": "9 AM - 3 PM"},
            {"name": "Dr. Sarah Lee", "exp_yrs": 14, "rating": 4.7, "fee": 250, "dist_km": 12.5, "address": "Brooklyn, NY", "time": "10 AM - 4 PM"},
            {"name": "Dr. Michael Chen", "exp_yrs": 8, "rating": 4.4, "fee": 150, "dist_km": 8.1, "address": "Queens, NY", "time": "1 PM - 7 PM"}
        ],
        "🩸 Diabetes": [
            {"name": "Dr. Emily Davis", "exp_yrs": 19, "rating": 4.8, "fee": 300, "dist_km": 4.5, "address": "Upper East Side, NY", "time": "8 AM - 2 PM"},
            {"name": "Dr. Robert Taylor", "exp_yrs": 11, "rating": 4.5, "fee": 200, "dist_km": 15.0, "address": "Staten Island, NY", "time": "12 PM - 6 PM"}
        ],
        "🫘 Chronic Kidney Disease": [
            {"name": "Dr. William Martinez", "exp_yrs": 30, "rating": 4.9, "fee": 450, "dist_km": 6.0, "address": "Midtown, NY", "time": "10 AM - 5 PM"},
            {"name": "Dr. Laura White", "exp_yrs": 16, "rating": 4.6, "fee": 280, "dist_km": 11.2, "address": "Bronx, NY", "time": "9 AM - 1 PM"}
        ]
    },
    "Pune": {
         "❤️ Heart Disease": [
            {"name": "Dr. S. Patil", "exp_yrs": 12, "rating": 4.7, "fee": 700, "dist_km": 5.2, "address": "Kothrud, Pune", "time": "9 AM - 1 PM"},
            {"name": "Dr. Amit Deshmukh", "exp_yrs": 22, "rating": 4.9, "fee": 1500, "dist_km": 12.0, "address": "Baner, Pune", "time": "10 AM - 5 PM"},
            {"name": "Dr. Neha Sharma", "exp_yrs": 8, "rating": 4.5, "fee": 500, "dist_km": 2.1, "address": "Shivajinagar, Pune", "time": "4 PM - 8 PM"}
        ],
        "🩸 Diabetes": [
            {"name": "Dr. Priya Kadam", "exp_yrs": 15, "rating": 4.6, "fee": 900, "dist_km": 6.0, "address": "Hinjewadi, Pune", "time": "5 PM - 9 PM"},
            {"name": "Dr. R. K. Joshi", "exp_yrs": 18, "rating": 4.8, "fee": 1200, "dist_km": 8.5, "address": "Viman Nagar, Pune", "time": "11 AM - 3 PM"},
            {"name": "Dr. Aniket More", "exp_yrs": 6, "rating": 4.2, "fee": 400, "dist_km": 14.1, "address": "Wakad, Pune", "time": "10 AM - 2 PM"}
        ],
        "🫘 Chronic Kidney Disease": [
            {"name": "Dr. V. Kulkarni", "exp_yrs": 25, "rating": 4.9, "fee": 1800, "dist_km": 4.4, "address": "Deccan Gymkhana, Pune", "time": "10 AM - 4 PM"},
            {"name": "Dr. Shruti Awate", "exp_yrs": 13, "rating": 4.7, "fee": 1000, "dist_km": 9.9, "address": "Kharadi, Pune", "time": "4 PM - 8 PM"}
        ]
    }
}

def get_user_location():
    """
    Get user's location using browser's Geolocation API via custom component.
    Falls back to IP-based detection if geolocation fails.
    Returns: (detected_city, location_data) tuple or (None, None)
    """
    html_code = """
    <script>
    const getLocation = () => {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    const lat = position.coords.latitude;
                    const lng = position.coords.longitude;
                    window.parent.postMessage({
                        type: "streamlit:setComponentValue",
                        value: {lat: parseFloat(lat.toFixed(4)), lng: parseFloat(lng.toFixed(4))}
                    }, "*");
                },
                (error) => {
                    // Fallback to IP-based detection if geolocation denied
                    console.log("Geolocation denied, falling back to IP detection");
                    window.parent.postMessage({
                        type: "streamlit:setComponentValue",
                        value: null
                    }, "*");
                }
            );
        } else {
            window.parent.postMessage({
                type: "streamlit:setComponentValue",
                value: null
            }, "*");
        }
    };
    
    getLocation();
    </script>
    <div style="padding: 8px; text-align: center; color: #999; font-size: 11px; animation: pulse 1.5s ease-in-out infinite;">
        📍 Requesting access to your location...
    </div>
    <style>
    @keyframes pulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }
    </style>
    """
    
    location_data = components.html(html_code, height=45)
    
    # If HTML component returns coordinates, map to nearest city
    if location_data and isinstance(location_data, dict) and 'lat' in location_data:
        detected_city, distance_km = get_nearest_city(location_data['lat'], location_data['lng'])
        if detected_city:
            location_data['method'] = "Browser Geolocation"
            location_data['distance_km'] = distance_km
            return detected_city, location_data
        else:
            # Try reverse geocoding as additional check
            reverse_city = reverse_geocode_coordinates(location_data['lat'], location_data['lng'])
            if reverse_city:
                detected_city, _ = get_nearest_city(location_data['lat'], location_data['lng'])
                if detected_city:
                    location_data['method'] = "Browser Geolocation + Reverse Geocoding"
                    return detected_city, location_data
    
    # Fallback to IP-based detection
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5).json()
        city = response.get("city")
        if city:
            return city, {"method": "IP-based (fallback)", "ip": response.get("query")}
    except:
        pass
    
    return None, None

def rank_doctors(doctor_list, top_n=3):
    """ML Multi-Criteria Decision Analysis using Min-Max Normalization."""
    if not doctor_list:
        return pd.DataFrame()
        
    df = pd.DataFrame(doctor_list)
    
    df['n_rating'] = (df['rating'] - df['rating'].min()) / (df['rating'].max() - df['rating'].min() + 0.001)
    df['n_exp'] = (df['exp_yrs'] - df['exp_yrs'].min()) / (df['exp_yrs'].max() - df['exp_yrs'].min() + 0.001)
    df['n_fee'] = 1 - ((df['fee'] - df['fee'].min()) / (df['fee'].max() - df['fee'].min() + 0.001))
    df['n_dist'] = 1 - ((df['dist_km'] - df['dist_km'].min()) / (df['dist_km'].max() - df['dist_km'].min() + 0.001))
    
    df['ml_score'] = (df['n_rating'] * 0.40) + (df['n_exp'] * 0.30) + (df['n_fee'] * 0.15) + (df['n_dist'] * 0.15)
    df['match_pct'] = (df['ml_score'] * 100).round(1)
    
    return df.sort_values(by='ml_score', ascending=False).head(top_n)

def send_gmail_report(user_email, top_doctors_df, disease, patient_id, city):
    """Sends the formatted HTML report via SMTP."""
    SENDER_EMAIL = "royvibhor082@gmail.com" 
    SENDER_APP_PASSWORD = "rfgf xxiw egjk mnfg" 
    
    msg = MIMEMultipart("alternative")
    msg['Subject'] = f"PULSE Medical Alert & Doctor Recommendations - {patient_id}"
    msg['From'] = SENDER_EMAIL
    msg['To'] = user_email

    html = f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #333;">
        <h2 style="color: #00d4ff; background-color: #0f172a; padding: 10px;">PULSE System - Automated Clinical Report</h2>
        <p>Based on your recent assessment for <strong>{disease}</strong>, our AI ranking system has identified the top specialists near your detected location in <strong>{city}</strong>.</p>
        <hr>
        <h3>Top Recommended Specialists:</h3>
    """
    
    for _, doc in top_doctors_df.iterrows():
        currency = "$" if city == "New York" else "₹"
        html += f"""
        <div style="background-color: #f8fafc; padding: 15px; margin-bottom: 10px; border-left: 5px solid #00d4ff;">
            <h4 style="margin: 0; color: #0f172a;">Dr. {doc['name']} <span style="color: #4ade80;">({doc['match_pct']}% Match)</span></h4>
            <ul style="margin-top: 10px;">
                <li><strong>Rating:</strong> {doc['rating']}/5.0</li>
                <li><strong>Experience:</strong> {doc['exp_yrs']} Years</li>
                <li><strong>Consultation Fee:</strong> {currency}{doc['fee']}</li>
                <li><strong>Distance:</strong> {doc['dist_km']} km</li>
                <li><strong>Clinic:</strong> {doc['address']}</li>
                <li><strong>Timings:</strong> {doc['time']}</li>
            </ul>
        </div>
        """
        
    html += """
        <p style="font-size: 12px; color: #777; margin-top: 20px;">
           <em>Disclaimer: PULSE is a screening tool. This is an automated email generated by the PULSE Final Year Project System.</em>
        </p>
      </body>
    </html>
    """
    
    msg.attach(MIMEText(html, "html"))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, user_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Email Error: {e}")
        return False

# ==========================================
# 🟢 8. SIDEBAR & NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("<div style='text-align: center; padding: 20px;'><img src='https://cdn-icons-png.flaticon.com/512/2966/2966327.png' width='100' style='border-radius: 50%; background: rgba(255,255,255,0.1); padding: 15px;'></div>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: white; font-size: 2em; margin-top: 0;'>PULSE System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.8); font-size: 0.9em;'><i>Predictive Unified Learning System for Health Evaluation</i></p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("<p style='color: white; font-weight: 600; font-size: 1.1em;'>📍 Navigation Menu</p>", unsafe_allow_html=True)
    page = st.radio("Select Module:", ["🩺 Risk Assessment", "📈 Patient History Tracker", "📖 Clinical Reference", "ℹ️ About PULSE"], label_visibility="collapsed")
    st.markdown("---")
    
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #00d4ff; margin-top: 30px;'>
        <p style='color: white; font-weight: 600; margin: 0 0 8px 0;'>⚠️ Important Notice</p>
        <p style='color: rgba(255,255,255,0.9); font-size: 0.85em; margin: 0;'>
            <strong>PULSE</strong> is a <strong>screening tool only</strong>. 
            It is <strong>NOT</strong> a diagnostic device. 
            Always consult a qualified healthcare professional for medical advice.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='color: rgba(255,255,255,0.7); font-size: 0.8em; text-align: center; margin-top: 40px;'>
        <p>Made with ❤️ for Healthcare</p>
        <p>Version 1.0 | Final Year Project</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 🟢 PAGE 1: RISK ASSESSMENT
# ==========================================
if page == "🩺 Risk Assessment":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0099cc 0%, #00d4ff 100%); padding: 40px 30px; border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='color: #0f172a; text-align: center; margin: 0; text-shadow: 0 2px 10px rgba(0,0,0,0.2); font-size: 2.5em;'>
            🩺 PULSE Risk Assessment System
        </h1>
        <p style='color: #0f172a; text-align: center; font-size: 1.1em; margin-top: 10px; font-weight: 600;'>
            Advanced Medical Screening & Risk Evaluation
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected_disease = st.selectbox(
            "Select Disease Module:",
            ["❤️ Heart Disease", "🩸 Diabetes", "🫘 Chronic Kidney Disease"],
            label_visibility="visible"
        )
    
    st.markdown("---")
    
    patient_id = st.text_input(
        "Patient ID / Name (For Database Logging):",
        "PATIENT-001",
        help="Enter a unique identifier for this patient record"
    )

    # -------------------------------------------------------
    # ✅ FIXED FORM — ONE submit button at the bottom
    # -------------------------------------------------------
    with st.form("pulse_form"):
        st.markdown("<h3 style='color: #00ffff; border-bottom: 3px solid #00d4ff; padding-bottom: 10px;'>📋 Patient Physiological Parameters</h3>", unsafe_allow_html=True)
        
        raw_inputs = {}  # initialise so it always exists

        # ---------------------------------------------------------
        # HEART DISEASE FORM
        # ---------------------------------------------------------
        if selected_disease == "❤️ Heart Disease":
            if heart_model is None:
                st.error("❌ Heart Disease model not found. Run `pulse_heart_engine.py` first.")
                st.stop()

            st.markdown("<p style='color: #00ffff; font-weight: 600; margin-top: 20px; font-size: 1.1em;'>💙 Basic Cardiac Parameters</p>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age (years)", 20, 100, 50)
                sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
                trestbps = st.number_input("Resting BP (mmHg)", 80, 220, 120)
            with c2:
                chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
                fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
            with c3:
                cp = st.selectbox("Chest Pain Type", [0,1,2,3], help="0=Typical Angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic")
                exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)

            st.markdown("<p style='color: #00ffff; font-weight: 600; margin-top: 20px; font-size: 1.1em;'>🔬 Advanced Cardiac Indicators</p>", unsafe_allow_html=True)
            c4, c5 = st.columns(2)
            with c4:
                restecg = st.selectbox("Resting ECG", [0,1,2])
                slope = st.selectbox("Slope of ST Segment", [0,1,2])
            with c5:
                ca = st.selectbox("Major Vessels (Fluoroscopy)", [0,1,2,3])
                thal = st.selectbox("Thalassemia", [1,2,3])

            # store inputs so they are available after submit
            raw_inputs = {
                'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
                'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
                'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
            }

        # ---------------------------------------------------------
        # DIABETES FORM
        # ---------------------------------------------------------
        elif selected_disease == "🩸 Diabetes":
            if diab_model is None:
                st.error("❌ Diabetes model not found. Run `pulse_diabetes_engine.py` first.")
                st.stop()

            st.markdown("<p style='color: #00ffff; font-weight: 600; margin-top: 20px; font-size: 1.1em;'>🩺 Metabolic & Glucose Parameters</p>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age (years)", 20, 100, 45)
                glucose = st.number_input("Plasma Glucose", 50, 300, 110)
                bp = st.number_input("Diastolic BP (mmHg)", 40, 150, 75)
            with c2:
                bmi = st.number_input("BMI (Body Mass Index)", 10.0, 60.0, 25.0, step=0.1)
                insulin = st.number_input("Serum Insulin (mu U/ml)", 0, 800, 80)
            with c3:
                pregnancies = st.number_input("Pregnancies", 0, 20, 0)
                skin = st.number_input("Triceps Skin Fold (mm)", 0, 100, 20)
                pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.4, step=0.05)

            raw_inputs = {
                'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': bp,
                'SkinThickness': skin, 'Insulin': insulin, 'BMI': bmi,
                'DiabetesPedigreeFunction': pedigree, 'Age': age
            }

        # ---------------------------------------------------------
        # CKD FORM
        # ---------------------------------------------------------
        elif selected_disease == "🫘 Chronic Kidney Disease":
            if ckd_model is None:
                st.error("❌ CKD model not found. Run `pulse_ckd_engine.py` first.")
                st.stop()

            st.markdown("<p style='color: #00ffff; font-weight: 600; margin-top: 20px; font-size: 1.1em;'>🫘 Renal Function & Blood Chemistry</p>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age (years)", 20, 100, 60)
                sys_bp = st.number_input("Systolic BP (mmHg)", 80, 250, 130)
                dias_bp = st.number_input("Diastolic BP (mmHg)", 40, 150, 80)
                gfr = st.number_input("Glomerular Filtration Rate (GFR)", 5.0, 150.0, 90.0)
            with c2:
                creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.1, 15.0, 0.9, step=0.1)
                bun = st.number_input("BUN Levels (mg/dL)", 2.0, 100.0, 15.0)
                hba1c = st.number_input("HbA1c (%)", 4.0, 15.0, 5.5, step=0.1)
                protein = st.selectbox("Protein in Urine", [0, 1, 2, 3, 4, 5])
            with c3:
                bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
                sodium = st.number_input("Serum Sodium (mEq/L)", 110.0, 160.0, 140.0)
                hemo = st.number_input("Hemoglobin Levels (g/dL)", 5.0, 20.0, 14.0)
                fatigue = st.selectbox("Fatigue Levels", [0, 1, 2, 3, 4, 5])

            raw_inputs = {
                'Age': age, 'SystolicBP': sys_bp, 'DiastolicBP': dias_bp, 'GFR': gfr,
                'SerumCreatinine': creatinine, 'BUNLevels': bun, 'HbA1c': hba1c,
                'ProteinInUrine': protein, 'BMI': bmi, 'SerumElectrolytesSodium': sodium,
                'HemoglobinLevels': hemo, 'FatigueLevels': fatigue
            }

        # ---------------------------------------------------------
        # ✅ ONE SINGLE SUBMIT BUTTON FOR ALL THREE DISEASE FORMS
        # ---------------------------------------------------------
        st.markdown("---")
        submitted = st.form_submit_button("🚀 Run PULSE Analysis", use_container_width=True)

    # ---------------------------------------------------------
    # PROCESSING LOGIC & DISPLAY RESULTS (STATE MANAGED)
    # ---------------------------------------------------------
    if submitted and raw_inputs:
        st.session_state.assessment_complete = True
        st.session_state.patient_id = patient_id
        st.session_state.selected_disease = selected_disease
        st.session_state.alerts = check_clinical_alerts(selected_disease, raw_inputs)

        with st.spinner("Running PULSE Engine..."):
            df_input = pd.DataFrame([raw_inputs])
            
            if selected_disease == "❤️ Heart Disease":
                df_input['age_group'] = pd.cut(df_input['age'], bins=[0, 40, 50, 60, 70, 100], labels=[0, 1, 2, 3, 4]).astype(int)
                df_input['high_bp'] = (df_input['trestbps'] > 130).astype(int)
                df_input['high_chol'] = (df_input['chol'] > 240).astype(int)
                df_input['low_thalach'] = (df_input['thalach'] < 120).astype(int)
                df_input['oldpeak_cat'] = pd.cut(df_input['oldpeak'], bins=[-0.1, 0, 1, 2, 10], labels=[0, 1, 2, 3]).astype(int)
                
                for col in heart_features:
                    if col not in df_input.columns: df_input[col] = 0
                X_live = df_input[heart_features]
                X_scaled = heart_scaler.transform(X_live)
                prob = heart_model.predict_proba(X_scaled)[0][1] * 100
                st.session_state.result = _build_heart_recommendation(prob, raw_inputs)
                model_used, features_used = heart_model, heart_features

            elif selected_disease == "🩸 Diabetes":
                df_input['Age_Group'] = pd.cut(df_input['Age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(int)
                df_input['BMI_Class'] = pd.cut(df_input['BMI'], bins=[0, 25, 30, 35, 100], labels=[0, 1, 2, 3]).astype(int)
                df_input['High_Glucose'] = (df_input['Glucose'] > 140).astype(int)
                df_input['High_BP'] = (df_input['BloodPressure'] > 80).astype(int)
                
                for col in diab_features:
                    if col not in df_input.columns: df_input[col] = 0
                X_live = df_input[diab_features]
                X_scaled = diab_scaler.transform(X_live)
                prob = diab_model.predict_proba(X_scaled)[0][1] * 100
                st.session_state.result = _build_diabetes_recommendation(prob, raw_inputs)
                model_used, features_used = diab_model, diab_features

            elif selected_disease == "🫘 Chronic Kidney Disease":
                df_input['GFR_Stage'] = pd.cut(df_input['GFR'], bins=[-1, 15, 30, 60, 90, 200], labels=[5, 4, 3, 2, 1]).astype(int)
                df_input['High_Creatinine'] = (df_input['SerumCreatinine'] > 1.2).astype(int)
                df_input['High_BUN'] = (df_input['BUNLevels'] > 20).astype(int)
                df_input['High_BP_Sys'] = (df_input['SystolicBP'] > 130).astype(int)
                
                for col in ckd_features:
                    if col not in df_input.columns: df_input[col] = 0
                X_live = df_input[ckd_features]
                X_scaled = ckd_scaler.transform(X_live)
                prob = ckd_model.predict_proba(X_scaled)[0][1] * 100
                st.session_state.result = _build_ckd_recommendation(prob, raw_inputs)
                model_used, features_used = ckd_model, ckd_features

            # Pre-compute SHAP values
            try:
                explainer = shap.Explainer(model_used)
                shap_values = explainer(X_scaled)
                
                if isinstance(explainer.expected_value, (list, np.ndarray)):
                    shap_vals = shap_values.values[0, :, 1]
                else:
                    shap_vals = shap_values.values[0]

                shap_df = pd.DataFrame({
                    'Feature': features_used,
                    'Patient_Value': X_live.iloc[0].values,
                    'Impact': shap_vals
                })
                
                shap_df['Abs_Impact'] = shap_df['Impact'].abs()
                shap_df = shap_df.sort_values(by='Abs_Impact', ascending=False)
                
                st.session_state.risk_drivers = shap_df[shap_df['Impact'] > 0].head(4)
                st.session_state.protectors = shap_df[shap_df['Impact'] < 0].head(4)
            except Exception:
                st.session_state.risk_drivers = None
                st.session_state.protectors = None

        save_patient_record(patient_id, selected_disease, prob)
        st.toast(f"✅ Record saved for {patient_id}")

    # ==========================================
    # 🟢 RENDER DASHBOARD
    # ==========================================
    if st.session_state.get('assessment_complete', False):
        
        for alert in st.session_state.alerts:
            st.markdown(f"<div class='alert-box'>🚨 <b>{alert}</b></div>", unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #0099cc 0%, #00d4ff 100%); padding: 30px; border-radius: 15px; margin-bottom: 20px;'>
            <h2 style='color: #0f172a; margin: 0; text-align: center; font-size: 2em;'>📋 PULSE Diagnostic Report</h2>
            <p style='color: #0f172a; text-align: center; margin-top: 10px; font-weight: 600;'>Comprehensive Risk Assessment & Clinical Insights</p>
        </div>
        """, unsafe_allow_html=True)
        
        result = st.session_state.result
        
        tab_report, tab_shap, tab_doctors = st.tabs([
            "📑 Clinical Assessment", 
            "🔍 AI Risk Breakdown", 
            "🏥 AI Specialists & Email Dispatch"
        ])

        with tab_report:
            col_left, col_right = st.columns([1, 2])

            with col_left:
                st.markdown("<h3 style='color: #00ffff; border-bottom: 3px solid #00d4ff; padding-bottom: 10px;'>📊 Risk Score</h3>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #0f172a 0%, #1a2744 100%); 
                            padding: 25px; border-radius: 15px; border: 2px solid #00d4ff; text-align: center;'>
                    <p style='margin: 0; color: #e0f2fe; font-size: 0.9em; font-weight: 600;'>Calculated Risk Probability</p>
                    <h1 style='margin: 15px 0; color: #00ffff; font-size: 3em; background: none; -webkit-text-fill-color: unset; text-shadow: 0 0 20px rgba(0, 212, 255, 0.4);'>
                        {result['risk_pct']:.1f}%
                    </h1>
                </div>
                """, unsafe_allow_html=True)
                
                if "LOW RISK" in result['risk_label']: st.markdown(f"<div class='risk-low'>{result['risk_label']}</div>", unsafe_allow_html=True)
                elif "MEDIUM RISK" in result['risk_label']: st.markdown(f"<div class='risk-med'>{result['risk_label']}</div>", unsafe_allow_html=True)
                else: st.markdown(f"<div class='risk-high'>{result['risk_label']}</div>", unsafe_allow_html=True)
                
                st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
                st.progress(min(result['risk_pct'] / 100, 1.0))
                st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

                pdf_bytes = create_pdf_report(
                    st.session_state.patient_id,
                    st.session_state.selected_disease,
                    result['risk_pct'],
                    result['risk_label'],
                    result['advice_summary'],
                    result['advice_detail']
                )
                st.download_button(
                    label="📥 Download Clinical PDF Report",
                    data=pdf_bytes,
                    file_name=f"PULSE_Report_{st.session_state.patient_id}.pdf",
                    mime="application/pdf",
                    key="pdf_download_btn"
                )

            with col_right:
                st.markdown("<h3 style='color: #00ffff; border-bottom: 3px solid #00d4ff; padding-bottom: 10px;'>💡 Actionable Intelligence</h3>", unsafe_allow_html=True)
                st.info(result['advice_summary'])
                st.markdown("---")
                for category, tips in result['advice_detail'].items():
                    with st.expander(f"**{category}**", expanded=False):
                        for tip in tips: st.write(f"• {tip}")

        with tab_shap:
            st.markdown("<h3 style='color: #00ffff; border-bottom: 3px solid #00d4ff; padding-bottom: 10px;'>🔍 Patient-Specific Risk Breakdown</h3>", unsafe_allow_html=True)
            st.write("Based on our AI analysis, here are the specific factors driving this patient's risk score.")
            
            if st.session_state.get('risk_drivers') is not None:
                c_risk, c_safe = st.columns(2)
                with c_risk:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%); padding: 20px; 
                                border-radius: 12px; border-left: 5px solid #ff6b6b; margin-bottom: 15px;'>
                        <p style='color: #fecaca; font-weight: 600; margin: 0 0 5px 0;'>🚨 Top Risk Drivers</p>
                        <p style='color: #fed7aa; font-size: 0.9em; margin: 0;'><i>Factors pushing the risk HIGHER</i></p>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.session_state.risk_drivers.empty:
                        st.write("✓ No major risk drivers detected.")
                    else:
                        for _, row in st.session_state.risk_drivers.iterrows():
                            st.write(f"🔺 **{row['Feature']}** (Value: {row['Patient_Value']:.2f})")

                with c_safe:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #134e4a 0%, #0d3830 100%); padding: 20px; 
                                border-radius: 12px; border-left: 5px solid #4ade80; margin-bottom: 15px;'>
                        <p style='color: #d1fae5; font-weight: 600; margin: 0 0 5px 0;'>🛡️ Top Protective Factors</p>
                        <p style='color: #a7f3d0; font-size: 0.9em; margin: 0;'><i>Factors keeping the risk LOWER</i></p>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.session_state.protectors.empty:
                        st.write("✓ No major protective factors detected.")
                    else:
                        for _, row in st.session_state.protectors.iterrows():
                            st.write(f"🔽 **{row['Feature']}** (Value: {row['Patient_Value']:.2f})")
            else:
                st.warning("⚠️ Risk breakdown is currently optimized for Tree-based models.")

        with tab_doctors:
            st.markdown("<h3 style='color: #00ffff; border-bottom: 3px solid #00d4ff; padding-bottom: 10px;'>🏥 AI Specialist Recommendation</h3>", unsafe_allow_html=True)
            st.write("Our system locates you via IP, cross-references our backend database, and uses Machine Learning to rank the best specialists based on rating, experience, proximity, and cost.")

            if "top_docs_df" not in st.session_state:
                st.session_state.top_docs_df = None
            if "detected_city" not in st.session_state:
                st.session_state.detected_city = None

            # ✅ FIXED: unique key for this button
            if st.button("📍 Detect Location & Run ML Recommendation Engine", key="doctor_loc_btn_unique", use_container_width=True):
                with st.spinner("Acquiring GPS/IP coordinates and calculating match scores..."):
                    detected_city, location_data = get_user_location()
                    
                    if detected_city and detected_city in MOCK_DOCTORS:
                        st.session_state.detected_city = detected_city
                        available_docs = MOCK_DOCTORS[detected_city].get(st.session_state.selected_disease, [])
                        
                        if available_docs:
                            st.session_state.top_docs_df = rank_doctors(available_docs, top_n=3)
                            
                            # Create detailed success message
                            method = location_data.get("method", "Unknown") if location_data else "Unknown"
                            distance_km = location_data.get("distance_km", "N/A") if location_data else "N/A"
                            if isinstance(distance_km, float):
                                distance_km = f"{distance_km:.1f} km"
                            
                            success_msg = f"✅ **Location verified: {detected_city}**\n\n"
                            success_msg += f"📡 Detection Method: {method}\n"
                            if distance_km != "N/A":
                                success_msg += f"📍 Distance to city center: {distance_km}\n"
                            success_msg += f"🏥 Found {len(st.session_state.top_docs_df)} top specialists"
                            
                            st.success(success_msg)
                            st.info(f"💡 The AI ranking engine has matched your health profile ({st.session_state.selected_disease}) with the best-rated specialists near you, ranked by experience, ratings, fees, and proximity.")
                        else:
                            st.warning(f"Location verified as {detected_city}, but no specialists for {st.session_state.selected_disease} are available in this region yet.")
                            st.session_state.top_docs_df = None
                    else:
                        city_str = detected_city if detected_city else "Unknown"
                        st.error(f"📍 Location detected as **{city_str}**, but it is outside our current database coverage.\n\n**Supported regions:** Mumbai, Kolkata, Delhi, Pune, New York")
                        st.session_state.top_docs_df = None

            if st.session_state.top_docs_df is not None:
                currency_sym = "$" if st.session_state.detected_city == "New York" else "₹"
                
                for index, doc in st.session_state.top_docs_df.iterrows():
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #1a2744 0%, #0f172a 100%); padding: 15px; border-radius: 10px; border-left: 4px solid #4ade80; margin-bottom: 15px;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <h4 style='color: #e0f2fe; margin: 0;'>{doc['name']}</h4>
                            <span style='background-color: #134e4a; color: #4ade80; padding: 5px 10px; border-radius: 20px; font-weight: bold; font-size: 0.9em;'>
                                {doc['match_pct']}% Match
                            </span>
                        </div>
                        <p style='color: #cbd5e1; margin: 10px 0 0 0; font-size: 0.9em;'>
                            ⭐ {doc['rating']} | 💼 {doc['exp_yrs']} Yrs Exp | 💰 {currency_sym}{doc['fee']} | 📍 {doc['dist_km']} km
                        </p>
                        <p style='color: #94a3b8; margin: 5px 0 0 0; font-size: 0.85em;'>
                            🏥 {doc['address']} ⏱️ {doc['time']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("#### 📧 Send Report to Patient")
                user_email = st.text_input("Enter Patient Email Address:", key="email_input")
                
                if user_email:
                    # ✅ FIXED: unique key for this button
                    if st.button("Send Clinical Report & Referrals", type="primary", key="send_email_btn_unique"):
                        with st.spinner("Connecting to secure mail server..."):
                            success = send_gmail_report(
                                user_email, 
                                st.session_state.top_docs_df, 
                                st.session_state.selected_disease, 
                                st.session_state.patient_id,
                                st.session_state.detected_city
                            )
                            if success:
                                st.balloons()
                                st.success(f"Report successfully dispatched to {user_email}!")
                            else:
                                st.error("Failed to send email. Check your SMTP credentials and App Password.")

# ==========================================
# 🟢 PAGE 2: PATIENT HISTORY TRACKER
# ==========================================
elif page == "📈 Patient History Tracker":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0099cc 0%, #00d4ff 100%); padding: 40px 30px; border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='color: #0f172a; text-align: center; margin: 0; text-shadow: 0 2px 10px rgba(0,0,0,0.2); font-size: 2.5em;'>
            📈 Longitudinal Patient Tracking
        </h1>
        <p style='color: #0f172a; text-align: center; font-size: 1.1em; margin-top: 10px; font-weight: 600;'>
            Monitor Risk Progression Across Multiple Visits
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("Track how a patient's risk profile evolves across multiple visits over time.")
    
    conn = sqlite3.connect('pulse_patients.db')
    df_history = pd.read_sql_query("SELECT * FROM patient_history", conn)
    conn.close()
    
    if df_history.empty:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #0c4a6e 0%, #082f4f 100%); padding: 25px; 
                    border-radius: 12px; border-left: 5px solid #00d4ff;'>
            <p style='color: #e0f2fe; font-weight: 600; margin: 0;'>ℹ️ No Patient History Found</p>
            <p style='color: #cbd5e1; font-size: 0.95em; margin: 10px 0 0 0;'>
                Run some risk assessments first to start tracking patient data.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        patients = df_history['patient_id'].unique()
        selected_patient = st.selectbox("Search for Patient ID:", patients)
        
        patient_data = df_history[df_history['patient_id'] == selected_patient].copy()
        patient_data['timestamp'] = pd.to_datetime(patient_data['timestamp'])
        patient_data = patient_data.sort_values('timestamp')
        
        st.markdown(f"<h3 style='color: #00ffff; border-bottom: 3px solid #00d4ff; padding-bottom: 10px;'>📊 Risk Trajectory for {selected_patient}</h3>", unsafe_allow_html=True)
        
        chart_data = patient_data[['timestamp', 'risk_score', 'disease_module']].pivot(index='timestamp', columns='disease_module', values='risk_score')
        st.line_chart(chart_data)
        
        st.markdown("<h4 style='color: #00ffff; margin-top: 30px;'>📋 Detailed History Logs</h4>", unsafe_allow_html=True)
        st.dataframe(patient_data, use_container_width=True)

# ==========================================
# 🟢 PAGE 3 & 4: REFERENCE & ABOUT
# ==========================================
elif page == "📖 Clinical Reference":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0099cc 0%, #00d4ff 100%); padding: 40px 30px; border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='color: #0f172a; text-align: center; margin: 0; text-shadow: 0 2px 10px rgba(0,0,0,0.2); font-size: 2.5em;'>
            📖 Clinical Reference Guide
        </h1>
        <p style='color: #0f172a; text-align: center; font-size: 1.1em; margin-top: 10px; font-weight: 600;'>
            Comprehensive Parameter Documentation
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a2744 0%, #0f172a 100%); 
                padding: 30px; border-radius: 15px; border: 2px solid #00d4ff;'>
        <h3 style='color: #00ffff; margin-top: 0;'>📚 Parameter Reference Documentation</h3>
        <p style='color: #e0f2fe; font-size: 1.05em;'>
            This guide provides detailed information about all input parameters used in PULSE risk assessments. 
            Understanding these parameters is crucial for accurate data entry and interpretation of results.
        </p>
        <ul style='color: #cbd5e1; line-height: 1.8;'>
            <li><strong style='color: #00ffff;'>Age:</strong> Patient age in years (20-100)</li>
            <li><strong style='color: #00ffff;'>Blood Pressure:</strong> Measured in mmHg (millimeters of mercury)</li>
            <li><strong style='color: #00ffff;'>BMI:</strong> Body Mass Index calculated as weight(kg)/height(m)²</li>
            <li><strong style='color: #00ffff;'>Glucose Levels:</strong> Plasma glucose measured in mg/dL</li>
            <li><strong style='color: #00ffff;'>Cholesterol:</strong> Serum cholesterol in mg/dL</li>
            <li><strong style='color: #00ffff;'>GFR:</strong> Glomerular Filtration Rate in mL/min/1.73m²</li>
            <li><strong style='color: #00ffff;'>Creatinine:</strong> Serum creatinine measured in mg/dL</li>
            <li><strong style='color: #00ffff;'>HbA1c:</strong> Glycated hemoglobin percentage</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #134e4a 0%, #0d3830 100%); 
                    padding: 20px; border-radius: 12px; border-left: 5px solid #4ade80;'>
            <h4 style='color: #d1fae5; margin-top: 0;'>✅ Healthy Ranges</h4>
            <ul style='color: #a7f3d0; margin: 0;'>
                <li>Systolic BP: &lt; 120 mmHg</li>
                <li>BMI: 18.5 - 24.9</li>
                <li>Fasting Glucose: 70 - 100 mg/dL</li>
                <li>GFR: &gt; 90 mL/min/1.73m²</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #7f1d1d 0%, #4c0519 100%); 
                    padding: 20px; border-radius: 12px; border-left: 5px solid #ff6b6b;'>
            <h4 style='color: #fecaca; margin: 0;'>⚠️ At-Risk Ranges</h4>
            <ul style='color: #fed7aa; margin: 0;'>
                <li>Systolic BP: ≥ 140 mmHg</li>
                <li>BMI: ≥ 30</li>
                <li>Fasting Glucose: ≥ 126 mg/dL</li>
                <li>GFR: &lt; 60 mL/min/1.73m²</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

elif page == "ℹ️ About PULSE":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0099cc 0%, #00d4ff 100%); padding: 40px 30px; border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='color: #0f172a; text-align: center; margin: 0; text-shadow: 0 2px 10px rgba(0,0,0,0.2); font-size: 2.5em;'>
            ℹ️ About PULSE Project
        </h1>
        <p style='color: #0f172a; text-align: center; font-size: 1.1em; margin-top: 10px; font-weight: 600;'>
            Predictive Unified Learning System for Health Evaluation
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a2744 0%, #0f172a 100%); 
                padding: 35px; border-radius: 15px; border: 2px solid #00d4ff; margin-bottom: 30px;'>
        <h3 style='color: #00ffff; margin-top: 0; border-bottom: 3px solid #00d4ff; padding-bottom: 15px;'>🏥 Project Overview</h3>
        <p style='color: #e0f2fe; font-size: 1.05em; line-height: 1.8;'>
            <strong>PULSE (Predictive Unified Learning System for Health Evaluation)</strong> is an advanced 
            AI-powered clinical decision support system developed as a Final Year B.Tech Project. 
            The system utilizes machine learning algorithms to predict the risk of three major chronic diseases:
        </p>
        <ul style='color: #cbd5e1; font-size: 1em; line-height: 1.8;'>
            <li><strong style='color: #4ade80;'>❤️ Cardiovascular Disease:</strong> Predicts heart disease risk based on cardiac parameters</li>
            <li><strong style='color: #4ade80;'>🩸 Diabetes Mellitus:</strong> Identifies Type 2 diabetes risk using metabolic indicators</li>
            <li><strong style='color: #4ade80;'>🫘 Chronic Kidney Disease:</strong> Assesses renal function decline risk</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0d4f4a 0%, #0a3a35 100%); 
                padding: 35px; border-radius: 15px; border: 2px solid #4ade80; margin-bottom: 30px;'>
        <h3 style='color: #d1fae5; margin-top: 0; border-bottom: 3px solid #4ade80; padding-bottom: 15px;'>👥 Development Team</h3>
        <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;'>
            <div style='background: rgba(74, 222, 128, 0.1); padding: 20px; border-radius: 10px; border-left: 4px solid #4ade80;'>
                <p style='color: #d1fae5; font-weight: 700; margin: 0 0 5px 0; font-size: 1.1em;'>👨‍💻 Pranjal Sharma</p>
                <p style='color: #a7f3d0; margin: 0; font-size: 0.95em;'>Lead Developer</p>
            </div>
            <div style='background: rgba(74, 222, 128, 0.1); padding: 20px; border-radius: 10px; border-left: 4px solid #4ade80;'>
                <p style='color: #d1fae5; font-weight: 700; margin: 0 0 5px 0; font-size: 1.1em;'>👨‍💻 Vibhor Anshuman Roy</p>
                <p style='color: #a7f3d0; margin: 0; font-size: 0.95em;'>ML Engineer</p>
            </div>
            <div style='background: rgba(74, 222, 128, 0.1); padding: 20px; border-radius: 10px; border-left: 4px solid #4ade80;'>
                <p style='color: #d1fae5; font-weight: 700; margin: 0 0 5px 0; font-size: 1.1em;'>👨‍💻 Nagisetti Surya</p>
                <p style='color: #a7f3d0; margin: 0; font-size: 0.95em;'>Data Scientist</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #4a2a1a 0%, #2a1810 100%); 
                padding: 35px; border-radius: 15px; border: 2px solid #f59e0b;'>
        <h3 style='color: #fbbf24; margin-top: 0; border-bottom: 3px solid #f59e0b; padding-bottom: 15px;'>🎓 Academic Institution</h3>
        <p style='color: #fed7aa; font-size: 1.05em; font-weight: 600; margin-top: 0;'>
            Department of Electrical & Computer Engineering
        </p>
        <p style='color: #fbbf24; font-size: 1.1em; font-weight: 700;'>
            Bharati Vidyapeeth College of Engineering, Pune
        </p>
        <p style='color: #fed7aa; font-size: 0.95em; margin-top: 15px; line-height: 1.6;'>
            PULSE represents the culmination of rigorous academic research, 
            advanced machine learning implementations, and practical healthcare application design.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(74, 222, 128, 0.05) 100%); 
            padding: 20px; border-radius: 10px; text-align: center; margin-top: 40px; border: 1px solid #00d4ff;'>
    <p style='color: #e0f2fe; font-weight: 600; margin: 0;'>
        💙 PULSE — B.Tech Final Year Project | Bharati Vidyapeeth COE Pune
    </p>
    <p style='color: #00ffff; font-size: 0.9em; margin: 5px 0 0 0;'>
        Advanced Healthcare Technology for Better Diagnosis | Version 1.0
    </p>
</div>
""", unsafe_allow_html=True)
