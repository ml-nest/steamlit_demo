import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Function to create and train the model (or load if already exists)
def get_model():
    model_path = "loan_model.pkl"
    
    # Check if model already exists
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    # Create a sample dataset for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data
    age = np.random.randint(22, 65, n_samples)
    income = np.random.randint(20000, 200000, n_samples)
    loan_amount = np.random.randint(5000, 100000, n_samples)
    credit_score = np.random.randint(300, 850, n_samples)
    loan_term = np.random.choice([12, 24, 36, 48, 60], n_samples)
    employment_years = np.random.randint(0, 20, n_samples)
    
    # Create features dataframe
    data = pd.DataFrame({
        'Age': age,
        'Income': income,
        'LoanAmount': loan_amount,
        'CreditScore': credit_score,
        'LoanTerm': loan_term,
        'EmploymentYears': employment_years
    })
    
    # Create target variable with some realistic rules
    # Higher probability of approval for higher income, credit score, employment years
    # Lower probability for higher loan amounts relative to income
    approval_probability = (
        0.4 + 
        0.2 * (data['Income'] / 200000) +
        0.2 * (data['CreditScore'] - 300) / 550 +
        0.1 * (data['EmploymentYears'] / 20) -
        0.1 * (data['LoanAmount'] / data['Income'])
    )
    approval_probability = np.clip(approval_probability, 0.05, 0.95)
    approved = np.random.binomial(1, approval_probability)
    
    # Add target to dataframe
    data['Approved'] = approved
    
    # Split the data
    X = data.drop('Approved', axis=1)
    y = data['Approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model
