import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Smart Fertilizer Advisor",
    page_icon="chemical.png"
)
@st.cache_resource
def load_model():
    with open("fertilizer_model.pkl","rb") as f:
      model, label_encoders, feature_columns = pickle.load(f)
    return model, label_encoders, feature_columns
model, label_encoders, feature_columns = load_model()
st.title("Fertilizer Recommendation System using Machine Learning")
st.image("sandy.jpg", use_container_width=True)
st.divider()
st.markdown(
    """
    **AI-powered decsion support for sustainable Indian agricultre**
    Get data-driven fertilizer recommendation based on soil, crop and field conditions.
    """
)

st.sidebar.header("Input Field Conditions")

season = st.sidebar.selectbox(
 "Select Season",["Kharif","Rabi","Zaid"]
)

crop = st.sidebar.selectbox(
    "Select Crop Type",
    label_encoders["Crop_Type"].classes_
)

nitrogen = st.sidebar.number_input(
    "Nitrogen Level (N)",
    min_value=0,
    max_value=200,
    value=50
)

phosphorous = st.sidebar.number_input(
    "Phosphorous Level (N)",
    min_value=0,
    max_value=200,
    value=40
)

Potassium = st.sidebar.number_input(
    "Potassium Level (K)",
    min_value=0,
    max_value=200,
    value=50
)

soil_ph = st.sidebar.number_input(
    "Soil pH",
    min_value=4.0,
    max_value=9.0,
    value=6.5,
    step=0.1
)

soil_moisture = st.sidebar.number_input(
    "Soil moisture (%)",
    min_value=0,
    max_value=100,
    value=35
)

temperature = st.sidebar.number_input(
    "Temperature (℃)",
    min_value=0,
    max_value=50,
    value=28
)

humidity = st.sidebar.number_input(
    "Humidity (%)",
    min_value=0,
    max_value=100,
    value=60
)

rainfall = st.sidebar.number_input(
    "Rainfall (mm)" ,
    min_value = 0,
    max_value = 5000,
    value = 1000
)

input_data = {
        "Soil_Type": label_encoders["Soil_Type"].classes_[0],
        "Soil_pH": soil_ph,
        "Soil_Moisture": soil_moisture,
        "Organic_Carbon": 0.8,
        "Electrical_Conductivity": 1.2,
        "Nitrogen_Level": nitrogen,
        "Phosphorus_Level": phosphorous,
        "Potassium_Level": Potassium,
        "Temperature": temperature,
        "Humidity": humidity,
        "Rainfall": rainfall,
        "Crop_Type": crop,
        "Crop_Growth_Stage": label_encoders["Crop_Growth_Stage"].classes_[0],
        "Season":season,
        "Irrigation_Type": label_encoders["Irrigation_Type"].classes_[0],
        "Previous_Crop": label_encoders["Previous_Crop"].classes_[0],
        "Region": label_encoders["Region"].classes_[0],
        "Fertilizer_Used_Last_Season": 150,
        "Yield_Last_Season": 4
    }
BASE_FERTILIZER_SCIENCE = {
    "urea": "Urea (CO(NH₂)₂) – Nitrogen fertilizer",
    "dap": "Diammonium Phosphate ((NH₄)₂HPO₄) – Nitrogen & Phosphorus",
    "mop": "Muriate of Potash (KCl) – Potassium fertilizer",
    "npk": "NPK Compound Fertilizer – Balanced nutrients",
    "compost": "Compost – Organic fertilizer rich in organic matter",
    "fym": "Farmyard Manure – Organic fertilizer"
    }
def build_fertilizer_mapping_from_dataset(csv_path):
        """
        Reads the dataset, extracts unique fertilizer labels,
        and maps them to scientific names safely.
        """

        import pandas as pd

        df = pd.read_csv(csv_path)

        # Extract unique fertilizer labels
        unique_ferts = df["Recommended_Fertilizer"].unique()

        fertilizer_mapping = {}

        for fert in unique_ferts:
            key = fert.lower().strip()

            # Map using base knowledge, fallback if unknown
            fertilizer_mapping[fert] = BASE_FERTILIZER_SCIENCE.get(
                key,
                f"{fert} (Fertilizer)"
            )

        return fertilizer_mapping
FERTILIZER_DETAILS = build_fertilizer_mapping_from_dataset(
    "../data/fertilizer_recommendation.csv"
)

if st.button("Recommend Fertilizer"):
    st.divider()

    st.subheader("Field Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Crop", crop)
    with c2:
        st.metric("Season", season)
    with c3:
        st.metric("Soil pH", round(soil_ph, 2))
    with c4:
        st.metric("Rainfall (mm)", rainfall)
    st.divider()
    col1,col2=st.columns([1.2,1])
    input_df = pd.DataFrame([input_data])
    for column in input_df.columns:
        if column in label_encoders:
            input_df[column] = label_encoders[column].transform(input_df[column])
    input_df = input_df[feature_columns]
    probabilities = model.predict_proba(input_df)[0]
    top3_indices = np.argsort(probabilities)[-3:][::-1]
    fertilizer_encoder = label_encoders["Recommended_Fertilizer"]
    top3_fertilizers = fertilizer_encoder.inverse_transform(model.classes_[top3_indices])
    top3_probs = probabilities[top3_indices]

    with col1:
            st.subheader("🌾 Recommended Fertilizer")
            main_fert = top3_fertilizers[0]
            fert_description = FERTILIZER_DETAILS.get(main_fert,f"{main_fert} (Fertilizer)")
            st.success(
                    f"**{fert_description}**  \n"
                    f"Confidence: **{round(top3_probs[0]*100, 2)}%**"
                )

            st.markdown("Alternative options")
            for fert, prob in zip(top3_fertilizers[1:], top3_probs[1:]):
                    st.write(f"• **{FERTILIZER_DETAILS.get(fert,f"{fert} (Fertilizer)")}** — {round(prob*100, 2)}%")

    with col2:
        st.subheader("📊 Recommendation Confidence")

        confidence_df = pd.DataFrame({
            "Fertilizer": top3_fertilizers,
            "Confidence (%)": top3_probs * 100
        })

        st.bar_chart(
            confidence_df.set_index("Fertilizer")
        )

        st.subheader("Key factors influencing this recommendation")
        st.caption(
            "The model considers, soil nutrients, crop type and field conditions. "
            "The chart below highlights the most influential parameters."
        )
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature":feature_columns,
            "Importance":importances
        })
        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        ).head(8)

        st.bar_chart(
            importance_df.set_index("Feature")
        )
    st.divider()




st.caption(
    "Smart Fertiliser Recommendation System | Machine Learning for Sustainable Agriculture"
    )






