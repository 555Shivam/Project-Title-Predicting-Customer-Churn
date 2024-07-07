pip install pandas numpy matplotlib seaborn scikit-learn tensorflow flask fastapi uvicorn streamlit shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the data
data = pd.read_csv('/content/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Replace 'No internet service' with 'No' in specific columns
columns_with_no_internet = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]

for column in columns_with_no_internet:
    data[column] = data[column].replace('No internet service', 'No')

# Explicitly convert 'Yes'/'No' to 1/0
yes_no_columns = [
    'Partner', 'Dependents', 'PhoneService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn'
]

for column in yes_no_columns:
    data[column] = data[column].replace({'Yes': 1, 'No': 0})

# List of categorical columns based on the provided data
categorical_columns = ['customerID', 'gender', 'MultipleLines', 'InternetService', 'Contract', 'PaymentMethod']

# Method 1: Label Encoding (for ordinal or binary categories)
label_encoder_columns = ['gender']

# Apply Label Encoding
label_encoder = LabelEncoder()
for column in label_encoder_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Method 2: One-Hot Encoding (for nominal categories with more than two values)
one_hot_encoder_columns = ['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod']

# Apply One-Hot Encoding
data = pd.get_dummies(data, columns=one_hot_encoder_columns)

# Drop 'customerID' as it is not needed for the analysis
data.drop(['customerID'], axis=1, inplace=True)

# Ensure all columns are numeric
for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Handle missing values
data.fillna(data.mean(), inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Handle missing values (example: fill with mean)
data.fillna(data.mean(), inplace=True)

# Feature engineering (example: converting categorical variables to numerical)
data = pd.get_dummies(data)

# Separate features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import matplotlib.pyplot as plt
import seaborn as sns

# Visualize target distribution
sns.countplot(x='Churn', data=data)
plt.show()

# Visualize correlations
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f')
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Initialize models
rf_model = RandomForestClassifier()
gb_model = GradientBoostingClassifier()
nn_model = MLPClassifier()

# Train models
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
nn_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)
nn_pred = nn_model.predict(X_test)

# Evaluate models
print("Random Forest:\n", classification_report(y_test, rf_pred))
print("Gradient Boosting:\n", classification_report(y_test, gb_pred))
print("Neural Network:\n", classification_report(y_test, nn_pred))

from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
}

# Initialize Grid Search
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy')

# Perform Grid Search
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

import shap

# Explain model predictions using SHAP
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=X.columns)
