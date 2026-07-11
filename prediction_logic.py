import pickle
import pandas as pd
import numpy as np

with open("fertilizer_model.pkl","rb") as f:
    model, label_encoders, feature_columns = pickle.load(f)

def predict_top3(input_data):
    input_df = pd.DataFrame([input_data])
    SEASON_MAPPING = {
        "Kharif" : "Summer",
        "Rabi"   : "Winter",
        "Zaid"   : "Dry"
    }
    if "Season" in input_data:
        input_data["Season"] = SEASON_MAPPING[input_data["Season"]]

    for column in input_df.columns:
        if column in label_encoders:
            input_df[column] = label_encoders[column].transform(input_df[column])

    input_df = input_df[feature_columns]
    probabilities = model.predict_proba(input_df)[0]
    top3_indices = np.argsort(probabilities)[-3:][::-1]
    fertilizer_encoder = label_encoders["Recommended_Fertilizer"]
    top3_fertilizers = fertilizer_encoder.inverse_transform(model.classes_[top3_indices])
    top3_probabilities = probabilities[top3_indices]
    results = []

    for fert,prob in zip(top3_fertilizers, top3_probabilities):
        results.append((fert, round(prob*100,2)))

    return results

def get_feature_importance():
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
            "Feature" : feature_columns,
            "Importance" : importances
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    return importance_df

if __name__ == "__main__":

    # Example test input
    sample_input = {
        "Soil_Type": "Loamy",
        "Soil_pH": 6.5,
        "Soil_Moisture": 35,
        "Organic_Carbon": 0.8,
        "Electrical_Conductivity": 1.2,
        "Nitrogen_Level": 40,
        "Phosphorus_Level": 45,
        "Potassium_Level": 60,
        "Temperature": 28,
        "Humidity": 65,
        "Rainfall": 1500,
        "Crop_Type": "Rice",
        "Crop_Growth_Stage": "Vegetative",
        "Season": "Kharif",
        "Irrigation_Type": "Drip",
        "Previous_Crop": "Wheat",
        "Region": "South",
        "Fertilizer_Used_Last_Season": 150,
        "Yield_Last_Season": 4
    }

    results = predict_top3(sample_input)

    print("\nTop 3 Recommended Fertilizers:\n")
    for fert, prob in results:
        print(f"{fert} — {prob}%")

    print("\nFeature Importance:\n")
    print(get_feature_importance().head())
