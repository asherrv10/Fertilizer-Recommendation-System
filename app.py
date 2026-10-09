import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Smart Fertilizer Advisor",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .rec-box {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .rec-box h2 {
        color: white !important;
        margin-top: 0;
    }
    .video-btn {
        display: inline-block;
        background-color: #EF4444;
        color: white !important;
        padding: 8px 16px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "fertilizer_model.pkl")
    with open(model_path, "rb") as f:
        model, label_encoders, feature_columns = pickle.load(f)
    return model, label_encoders, feature_columns

try:
    model, label_encoders, feature_columns = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure train.py has been run to generate fertilizer_model.pkl.")
    st.stop()

BASE_FERTILIZER_SCIENCE = {
    "urea": "Urea (CO(NH₂)₂) – High Nitrogen fertilizer for vegetative growth",
    "dap": "Diammonium Phosphate ((NH₄)₂HPO₄) – High Nitrogen & Phosphorus content",
    "mop": "Muriate of Potash (KCl) – Essential Potassium fertilizer for crop quality & immunity",
    "ssp": "Single Super Phosphate – Rich in Phosphorus, Calcium & Sulphur",
    "10:26:26 npk": "NPK 10:26:26 – Balanced formulation for root development and flowering",
    "13:32:26 npk": "NPK 13:32:26 – High Phosphorus and Potassium booster",
    "12:32:16 npk": "NPK 12:32:16 – Balanced starter fertilizer",
    "50:26:26 npk": "NPK 50:26:26 – Concentrated Nitrogen and Potassium blend",
    "19:19:19 npk": "NPK 19:19:19 – All-purpose balanced nutrient spray/fertilizer",
    "18:46:00 npk": "NPK 18:46:00 – High analysis Phosphate fertilizer",
    "20:20:20 npk": "NPK 20:20:20 – General purpose balanced nutrient formulation",
    "10:10:10 npk": "NPK 10:10:10 – Equal ratio nutrient formulation",
    "magnesium sulphate": "Magnesium Sulphate (Epsom Salt) – Boosts chlorophyll formation",
    "chilated micronutrient": "Chelated Micronutrients – Corrects trace mineral deficiencies",
    "sulphur": "Elemental Sulphur – Improves soil pH and oilseed quality",
    "ammonium sulphate": "Ammonium Sulphate – Provides Nitrogen & Sulphur",
    "ferrous sulphate": "Ferrous Sulphate – Corrects Iron chlorosis and yellowing",
    "white potash": "White Potash (Sulphate of Potash) – Chloride-free Potassium source",
    "hydrated lime": "Hydrated Lime – Neutralizes acidic soils and supplies Calcium",
    "compost": "Compost – Organic fertilizer rich in soil organic matter",
    "fym": "Farmyard Manure – Traditional organic nutrient enrichment"
}

@st.cache_data
def build_fertilizer_details(csv_path="Crop and fertilizer dataset.csv"):
    full_path = os.path.join(BASE_DIR, csv_path)
    fertilizer_details = {}
    
    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
        fert_col = "Fertilizer" if "Fertilizer" in df.columns else "Recommended_Fertilizer"
        if fert_col in df.columns:
            for _, row in df.iterrows():
                fert = str(row[fert_col]).strip()
                link = str(row.get("Link", "")).strip() if "Link" in df.columns else ""
                key = fert.lower()
                desc = BASE_FERTILIZER_SCIENCE.get(key, f"{fert} – Customized Crop Nutrient")
                
                if fert not in fertilizer_details:
                    fertilizer_details[fert] = {
                        "description": desc,
                        "link": link if (link and link.startswith("http")) else ""
                    }
                elif link and not fertilizer_details[fert]["link"]:
                    fertilizer_details[fert]["link"] = link
                    
    return fertilizer_details

FERTILIZER_DETAILS = build_fertilizer_details()

# App Header
st.markdown('<div class="main-title">🌱 Smart Fertilizer Recommendation System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI-powered decision support for sustainable Indian agriculture. Get data-driven fertilizer recommendations based on soil, crop, and climatic conditions.</div>', 
    unsafe_allow_html=True
)
st.divider()

# Sidebar inputs
st.sidebar.header("📋 Input Field & Soil Conditions")

# Categorical Selectboxes
district_options = label_encoders["District_Name"].classes_ if "District_Name" in label_encoders else ["Kolhapur", "Solapur", "Satara", "Sangli", "Pune"]
district = st.sidebar.selectbox("Select District", district_options)

soil_color_options = label_encoders["Soil_color"].classes_ if "Soil_color" in label_encoders else ["Black", "Red", "Medium Brown", "Dark Brown", "Light Brown", "Reddish Brown"]
soil_color = st.sidebar.selectbox("Select Soil Color", soil_color_options)

crop_options = label_encoders["Crop"].classes_ if "Crop" in label_encoders else label_encoders.get("Crop_Type", label_encoders["Crop"]).classes_
crop = st.sidebar.selectbox("Select Crop Type", crop_options)

st.sidebar.subheader("🧪 Soil Nutrients & Parameters")
nitrogen = st.sidebar.number_input("Nitrogen Level (N)", min_value=0, max_value=250, value=80, step=5)
phosphorous = st.sidebar.number_input("Phosphorus Level (P)", min_value=0, max_value=250, value=50, step=5)
potassium = st.sidebar.number_input("Potassium Level (K)", min_value=0, max_value=250, value=100, step=5)
soil_ph = st.sidebar.number_input("Soil pH", min_value=3.0, max_value=10.0, value=6.5, step=0.1)

st.sidebar.subheader("🌤️ Environmental Conditions")
temperature = st.sidebar.number_input("Temperature (℃)", min_value=0, max_value=60, value=25, step=1)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0, max_value=5000, value=1000, step=50)

# Build input dictionary
input_data = {
    "District_Name": district,
    "Soil_color": soil_color,
    "Nitrogen": nitrogen,
    "Phosphorus": phosphorous,
    "Potassium": potassium,
    "pH": soil_ph,
    "Rainfall": rainfall,
    "Temperature": temperature,
    "Crop": crop
}

# Main content
st.subheader("🌾 Current Field Summary")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Crop", crop)
with c2:
    st.metric("District", district)
with c3:
    st.metric("Soil Color", soil_color)
with c4:
    st.metric("Soil pH", round(soil_ph, 2))
with c5:
    st.metric("N-P-K Ratio", f"{nitrogen}-{phosphorous}-{potassium}")

st.divider()

if st.button("🚀 Recommend Fertilizer", type="primary", use_container_width=True):
    # Preprocess inputs
    input_df = pd.DataFrame([input_data])
    
    for column in input_df.columns:
        if column in label_encoders:
            le = label_encoders[column]
            val = input_df[column].iloc[0]
            if isinstance(val, str):
                val_clean = val.strip()
                if val_clean in le.classes_:
                    input_df[column] = le.transform([val_clean])
                else:
                    input_df[column] = 0
            elif not isinstance(val, (int, float, np.integer, np.floating)):
                input_df[column] = le.transform(input_df[column])
                
    # Align feature ordering
    input_df = input_df[feature_columns]
    
    # Model inference
    probabilities = model.predict_proba(input_df)[0]
    top3_indices = np.argsort(probabilities)[-3:][::-1]
    
    fert_encoder = label_encoders.get("Fertilizer", label_encoders.get("Recommended_Fertilizer"))
    top3_fertilizers = fert_encoder.inverse_transform(model.classes_[top3_indices])
    top3_probs = probabilities[top3_indices]
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("🎯 Primary Recommendation")
        main_fert = top3_fertilizers[0]
        fert_info = FERTILIZER_DETAILS.get(main_fert, {})
        fert_desc = fert_info.get("description", BASE_FERTILIZER_SCIENCE.get(main_fert.lower(), f"{main_fert} (Fertilizer)"))
        video_link = fert_info.get("link", "")
        
        st.success(f"### 🥇 Recommended: **{main_fert}**\n\n**Confidence:** **{top3_probs[0]*100:.2f}%**\n\n{fert_desc}")
        
        if video_link:
            st.markdown(f'<a href="{video_link}" target="_blank" class="video-btn">▶️ Watch Application Video Guide</a>', unsafe_allow_html=True)
            st.write("")

        st.subheader("💡 Alternative Options")
        for fert, prob in zip(top3_fertilizers[1:], top3_probs[1:]):
            info = FERTILIZER_DETAILS.get(fert, {})
            desc = info.get("description", BASE_FERTILIZER_SCIENCE.get(fert.lower(), f"{fert} (Fertilizer)"))
            st.info(f"• **{fert}** — **{prob*100:.2f}% confidence**\n\n_{desc}_")
            
    with col2:
        st.subheader("📊 Recommendation Confidence")
        confidence_df = pd.DataFrame({
            "Fertilizer": top3_fertilizers,
            "Confidence (%)": top3_probs * 100
        })
        st.bar_chart(confidence_df.set_index("Fertilizer"))
        
        st.subheader("🔑 Key Influencing Parameters")
        st.caption("Feature importances learned by the trained Random Forest model.")
        
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance (%)": importances * 100
        }).sort_values(by="Importance (%)", ascending=False)
        
        st.bar_chart(importance_df.set_index("Feature"))

st.divider()
st.caption("Smart Fertilizer Recommendation System | Machine Learning for Sustainable Agriculture")
