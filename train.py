import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_and_save_model(csv_path="Crop and fertilizer dataset.csv", model_path="fertilizer_model.pkl"):
    # Determine absolute path if relative path is provided
    if not os.path.exists(csv_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "Crop and fertilizer dataset.csv")

    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Clean leading/trailing whitespaces in string columns
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'string':
            df[col] = df[col].astype(str).str.strip()

    feature_columns = ['District_Name', 'Soil_color', 'Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature', 'Crop']
    target_column = 'Fertilizer'

    # Fit label encoders for categorical columns
    label_encoders = {}
    categorical_cols = ['District_Name', 'Soil_color', 'Crop', target_column]

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Add aliases for backwards compatibility with any legacy key references
    label_encoders["Crop_Type"] = label_encoders["Crop"]
    label_encoders["Soil_Type"] = label_encoders["Soil_color"]
    label_encoders["Recommended_Fertilizer"] = label_encoders["Fertilizer"]

    X = df[feature_columns]
    y = df[target_column]

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Training Completed. Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoders[target_column].classes_))

    # Save trained model, label encoders, and feature list
    if not os.path.isabs(model_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, model_path)

    with open(model_path, "wb") as f:
        pickle.dump((model, label_encoders, feature_columns), f)
    
    print(f"Successfully saved trained model artifact to: {model_path}")
    return model, label_encoders, feature_columns

if __name__ == "__main__":
    train_and_save_model()
