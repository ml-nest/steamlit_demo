import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
import pickle
import os
from model import get_model

# Set page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="centered"
)

# Use cache_resource decorator to avoid retraining the model on each rerun
@st.cache_resource
def load_model():
    return get_model()

# Load or train the model
model = load_model()

# Title and description
st.title("Loan Approval Prediction")
st.markdown("""
This app predicts whether your loan application will be approved based on the information you provide.
""")

# Create sidebar for user inputs
st.sidebar.header("Enter Your Information")

# User input fields
age = st.sidebar.slider("Age", 18, 65, 30)
income = st.sidebar.slider("Annual Income ($)", 20000, 200000, 50000, step=1000)
loan_amount = st.sidebar.slider("Loan Amount ($)", 5000, 100000, 20000, step=1000)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
loan_term = st.sidebar.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
employment_years = st.sidebar.slider("Years of Employment", 0, 20, 3)

# Create a dataframe with user input
user_data = pd.DataFrame({
    'Age': [age],
    'Income': [income],
    'LoanAmount': [loan_amount],
    'CreditScore': [credit_score],
    'LoanTerm': [loan_term],
    'EmploymentYears': [employment_years]
})

# Main panel with prediction
st.header("Prediction")

# Display user inputs in the main panel
st.subheader("Your Information")
user_info_cols = st.columns(3)
with user_info_cols[0]:
    st.metric("Age", f"{age} years")
    st.metric("Loan Amount", f"${loan_amount:,}")
with user_info_cols[1]:
    st.metric("Annual Income", f"${income:,}")
    st.metric("Loan Term", f"{loan_term} months")
with user_info_cols[2]:
    st.metric("Credit Score", credit_score)
    st.metric("Employment", f"{employment_years} years")

# Make prediction when user clicks the button
if st.button("Predict Loan Approval"):
    # Make prediction
    prediction = model.predict(user_data)
    probability = model.predict_proba(user_data)
    
    # Display result
    st.subheader("Prediction Result")
    
    if prediction[0] == 1:
        st.success(f"Congratulations! Your loan is likely to be APPROVED with {probability[0][1]:.1%} confidence.")
        
        # Show approval factors
        st.subheader("Key Approval Factors")
        feature_importance = model.feature_importances_
        features = user_data.columns
        
        # Map feature importance to user values
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importance,
            'Your Value': [age, income, loan_amount, credit_score, loan_term, employment_years]
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Display top 3 factors
        top_factors = importance_df.head(3)
        st.write("The top factors influencing your approval:")
        for i, row in top_factors.iterrows():
            st.write(f"- **{row['Feature']}**: {row['Your Value']}")
    else:
        st.error(f"Unfortunately, your loan is likely to be DENIED with {probability[0][0]:.1%} confidence.")
        
        # Provide suggestions for improvement
        st.subheader("Suggestions for Improvement")
        
        if credit_score < 700:
            st.write("- Try to improve your credit score")
        
        if loan_amount > income * 0.5:
            st.write("- Consider requesting a lower loan amount")
        
        if employment_years < 2:
            st.write("- A longer employment history may help your application")

# Add feature explanation section
with st.expander("Understanding the Factors"):
    st.markdown("""
    ### Key Factors in Loan Approval
    
    - **Age**: Lenders consider age as a factor for stability and repayment capability.
    - **Income**: Higher income indicates better ability to repay the loan.
    - **Loan Amount**: The amount you're borrowing relative to your income affects risk assessment.
    - **Credit Score**: Higher scores indicate better credit history and lower risk.
    - **Loan Term**: The length of time for repayment affects monthly payments and overall interest.
    - **Employment Years**: Longer employment history suggests job stability.
    """)

# Footer
st.markdown("---")
st.markdown("### About This App")
st.markdown("""
This is a demonstration app using a simulated model. In real-world scenarios, financial institutions use more complex models with additional factors.
""")
