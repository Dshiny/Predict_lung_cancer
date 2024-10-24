# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:02:51 2024

@author: Dharshini
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv("C:/Users/Dharshini/Downloads/lung cancer data.csv")

# Feature engineering as before...
# (Add your feature engineering steps here)

# Convert categorical columns to numerical values
df["GENDER"].replace({"M": 1, "F": 0}, inplace=True)
df["LUNG_CANCER"].replace({"YES": 1, "NO": 0}, inplace=True)

# Extract target variable
y = df["LUNG_CANCER"]
x = pd.DataFrame(StandardScaler().fit_transform(df.drop(columns=["LUNG_CANCER"])), 
                 columns=df.drop(columns=["LUNG_CANCER"]).columns)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
x, y = smote.fit_resample(x, y)

# Initialize models
models = {
    "LogisticRegression": LogisticRegression(),
    "RandomForestClassifier": RandomForestClassifier(),
    "SupportVectorClassifier": SVC(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "BaggingClassifier": BaggingClassifier(),
    "AdaBoostClassifier": AdaBoostClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
}

# Set up hyperparameter grids
param_grids = {
    # Add hyperparameters for models (same as before or modified)
}

# Use Stratified K-Fold for better class distribution
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists for results
results = []
best_models = {}

# Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=42, stratify=y)

# Perform Grid Search with cross-validation for each model
for model_name, model_instance in models.items():
    print(f"Training {model_name}...")
    
    grid_search = GridSearchCV(estimator=model_instance,
                               param_grid=param_grids.get(model_name, {}),
                               scoring='accuracy',
                               cv=kf)
    
    grid_search.fit(x_train, y_train)
    best_models[model_name] = grid_search.best_estimator_

# Evaluate the best models on the test set
for model_name, model in best_models.items():
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append({
        "Model": model_name,
        "Accuracy": accuracy
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
print(results_df)

# Optional: Create a stacking model
estimators = [(name, model) for name, model in best_models.items()]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_model.fit(x_train, y_train)
stacking_accuracy = accuracy_score(y_test, stacking_model.predict(x_test))
print(f"Stacking Model Accuracy: {stacking_accuracy}")
