import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Load the data
df = pd.read_csv('Task 3 and 4_Loan_Data.csv')

# Prepare the features and target
X = df.drop(['customer_id', 'default'], axis=1)
y = df['default']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {'accuracy': accuracy, 'auc': auc}
    print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

# Choose the best model (based on AUC)
best_model_name = max(results, key=lambda x: results[x]['auc'])
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name}")

# Function to calculate expected loss
def calculate_expected_loss(loan_properties, recovery_rate=0.1):
    # Prepare the input data
    loan_df = pd.DataFrame([loan_properties])
    loan_scaled = scaler.transform(loan_df)
    
    # Predict probability of default
    pd = best_model.predict_proba(loan_scaled)[0, 1]
    
    # Calculate expected loss
    loan_amount = loan_properties['loan_amt_outstanding']
    expected_loss = loan_amount * pd * (1 - recovery_rate)
    
    return {
        'probability_of_default': pd,
        'expected_loss': expected_loss
    }

# Example usage
example_loan = {
    'credit_lines_outstanding': 3,
    'loan_amt_outstanding': 50000,
    'total_debt_outstanding': 75000,
    'income': 80000,
    'years_employed': 5,
    'fico_score': 700
}

result = calculate_expected_loss(example_loan)
print("\nExample loan prediction:")
print(f"Probability of Default: {result['probability_of_default']:.4f}")
print(f"Expected Loss: ${result['expected_loss']:.2f}")