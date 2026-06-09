import pandas as pd
import numpy as np
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_and_clean, get_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# load data
df = load_and_clean('data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

print("Labels found:", df['Label'].unique())
print("Shape:", df.shape)

# encode labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['Label'])

# get features
X, y_raw = get_features(df)
y = df['label_encoded']

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# train model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# save model files
joblib.dump(model, 'model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')
print("Model saved successfully!")
