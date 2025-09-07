# Loan Approval Prediction App

This is a simple Streamlit application that predicts whether a loan application will be approved based on various factors such as age, income, loan amount, credit score, loan term, and employment history.

## Features

- User-friendly interface for entering loan application details
- Real-time prediction of loan approval likelihood
- Explanations of important factors affecting the prediction
- Suggestions for improving loan approval chances

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Required packages: streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn

### Installation

1. Clone this repository or download the files
2. Install the required packages:
   ```
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn
   ```

### Running the Application

1. Navigate to the project directory
2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
3. The app will open in your default web browser at http://localhost:8501

## How to Use

1. Enter your information in the sidebar
2. Click the "Predict Loan Approval" button
3. View the prediction result and recommendations

## Model Information

The app uses a Random Forest classifier trained on synthetic data. The model considers the following factors:
- Age
- Annual Income
- Loan Amount
- Credit Score
- Loan Term (in months)
- Years of Employment

Note: This is a demonstration model and not intended for real financial decisions.

## Screenshots

![App Screenshot](screenshot.png)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
