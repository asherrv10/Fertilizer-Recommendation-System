import pickle
import pandas as pd
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fertilizer_model.pkl")

def load_model(model_path=MODEL_PATH):
    with open(model_path, "rb") as f:
        model, label_encoders, feature_columns = pickle.load(f)
    return model, label_encoders, feature_columns

model, label_encoders, feature_columns = load_model()

def predict_top3(input_data):
    """
    Given an input dict with feature keys (e.g. District_Name, Soil_color, Nitrogen, etc.),
    returns the top 3 recommended fertilizers with their confidence percentages.
    """
    input_df = pd.DataFrame([input_data])

    # Transform categorical feature inputs using stored label encoders
    for column in input_df.columns:
        if column in label_encoders:
            le = label_encoders[column]
            val = input_df[column].iloc[0]
            # Handle string vs integer encoded input
            if isinstance(val, str):
                val_clean = val.strip()
                if val_clean in le.classes_:
                    input_df[column] = le.transform([val_clean])
                else:
                    # Fallback to first class if unseen category
                    input_df[column] = 0
            elif not isinstance(val, (int, float, np.integer, np.floating)):
                input_df[column] = le.transform(input_df[column])

    # Ensure correct feature ordering
    input_df = input_df[feature_columns]

    probabilities = model.predict_proba(input_df)[0]
    top3_indices = np.argsort(probabilities)[-3:][::-1]
    
    # Get fertilizer encoder (supports both 'Fertilizer' and 'Recommended_Fertilizer' keys)
    fertilizer_encoder = label_encoders.get("Fertilizer", label_encoders.get("Recommended_Fertilizer"))
    
    top3_fertilizers = fertilizer_encoder.inverse_transform(model.classes_[top3_indices])
    top3_probabilities = probabilities[top3_indices]

    results = []
    for fert, prob in zip(top3_fertilizers, top3_probabilities):
        results.append((fert, round(prob * 100, 2)))

    return results

def get_feature_importance():
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    return importance_df

if __name__ == "__main__":
    sample_input = {
        "District_Name": "Kolhapur",
        "Soil_color": "Black",
        "Nitrogen": 80,
        "Phosphorus": 50,
        "Potassium": 100,
        "pH": 6.5,
        "Rainfall": 1000,
        "Temperature": 20,
        "Crop": "Sugarcane"
    }

    results = predict_top3(sample_input)

    print("\nTop 3 Recommended Fertilizers:\n")
    for fert, prob in results:
        print(f"• {fert} — {prob}%")

    print("\nFeature Importance:\n")
    print(get_feature_importance().head())
