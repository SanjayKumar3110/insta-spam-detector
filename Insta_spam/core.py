""" This core.py script is used to train and save model using the dataset instagram_comments.csv
    The final model is saved as insta_fake_detection.pkl file.
    The app.py is use the trained model for fake account detection."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv("instagram_comments.csv")

# If boolean-like columns are in string format, convert to numeric
binary_map = {'Yes': 1, 'No': 0, 'TRUE': 1, 'FALSE': 0, 'true': 1, 'false': 0}
columns_to_convert = ['profile pic', 'name==username', 'external URL', 'private']
for col in columns_to_convert:
    if df[col].dtype == object:
        df[col] = df[col].map(binary_map)

# Features and label
feature_columns = [
    'profile pic', 'nums/length username', 'fullname words', 'nums/length fullname',
    'name==username', 'description length', 'external URL', 'private',
    '#posts', '#followers', '#follows'
]

X = df[feature_columns]
y = df['fake']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("✅ Classification Report:\n")
print(classification_report(y_test, y_pred))

import joblib
joblib.dump(model, "instagram_fake_detection.pkl")
