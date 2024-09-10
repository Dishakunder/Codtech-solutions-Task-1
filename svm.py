import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Step 1: Define the dataset
data = {
    'Height': [170, 180, 160, 150, 175, 185, 165, 155, 172, 168, 182, 158,
               160, 165, 170, 175, 165, 155, 185, 172, 160, 155, 170,
               160, 150, 185, 160, 150, 175, 180, 170, 165, 160, 155, 180,
               185, 175, 155, 165, 175, 182, 172, 158, 145, 150, 155, 165],
    'Weight': [45, 50, 55, 60, 70, 85, 95, 110, 60, 90, 120, 48, 58, 72, 90, 85, 
               55, 45, 95, 75, 95, 105, 48, 52, 20, 50, 105, 95, 22, 90, 85, 
               120, 68, 54, 75, 92, 75, 54, 110, 70, 90, 55, 120, 45, 40, 48, 58],
    'Age': [25, 30, 22, 19, 28, 35, 45, 50, 20, 31, 40, 23, 27, 21, 32, 38,
            29, 25, 50, 33, 31, 19, 23, 22, 19, 50, 60, 21, 22, 31, 32, 24, 
            29, 21, 35, 27, 28, 31, 24, 30, 45, 21, 18, 23, 30, 29, 31],
    'Gender': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 
               'Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 
               'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female', 
               'Male', 'Female', 'Female', 'Female', 'Male', 'Male', 'Female', 
               'Male', 'Male', 'Female', 'Male', 'Female', 'Female', 'Male', 
               'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Female', 
               'Male', 'Female', 'Female', 'Male', 'Female']
}

# Step 2: Create the DataFrame
df = pd.DataFrame(data)

# Step 3: Calculate BMI
df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2

# Step 4: Function to classify BMI into categories
def classify_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 24.9:
        return "Normal weight"
    elif 25 <= bmi < 29.9:
        return "Overweight"
    elif 30 <= bmi < 34.9:
        return "Obese"
    elif 35 <= bmi < 39.9:
        return "Severely obese"
    else:
        return "Morbidly obese"

# Step 5: Apply the classification function to the BMI column
df['Category'] = df['BMI'].apply(classify_bmi)

# Step 6: Encode 'Gender' and 'Category'
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])  # Encode 'Female' as 0, 'Male' as 1

le_category = LabelEncoder()
df['Category'] = le_category.fit_transform(df['Category'])  # Encode BMI categories

# Step 7: Define features and target
X = df[['Height', 'Weight', 'Age', 'Gender']]
y = df['Category']

# Step 8: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 9: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 10: Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train_scaled, y_train)

# Best parameters from grid search
print(f"Best Parameters: {grid.best_params_}")

# Step 11: Evaluate the model with the best parameters
y_pred = grid.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy with tuned parameters: {acc:.2f}")

# Step 12: Save the best model, scaler, and encoders
joblib.dump(grid.best_estimator_, 'svm_bmi_model_tuned.pkl')
joblib.dump(scaler, 'svm_scaler_tuned.pkl')
joblib.dump(le_gender, 'svm_le_gender_tuned.pkl')
joblib.dump(le_category, 'svm_le_category_tuned.pkl')
