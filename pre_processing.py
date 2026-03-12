"""
Session 04 – Step 2: Preprocessing
Reads ingested data, splits, scales, and saves the preprocessor artifact.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess():
    os.makedirs("artifacts", exist_ok=True)
    # 1. Baca data Titanic
    df = pd.read_csv("train.csv")

    # 2. Feature Engineering (Pecah Cabin)
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Num'] = pd.to_numeric(df['Num'], errors='coerce')

    # 3. Tentukan Fitur (13 fitur)
    num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']
    cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    
    # Targetnya adalah 'Transported', BUKAN 'species'
    X = df[num_features + cat_features]
    y = df['Transported'].astype(int)

    # 4. Pipeline Preprocessing
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Simpan preprocessor ke artifacts
    joblib.dump(preprocessor, "artifacts/preprocessor.pkl")

    train_df = pd.concat([pd.DataFrame(X_train_proc), y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([pd.DataFrame(X_test_proc), y_test.reset_index(drop=True)], axis=1)
    
    print("✅ Preprocessing Selesai untuk Dataset Titanic.")
    return train_df, test_df

if __name__ == "__main__":
    preprocess()