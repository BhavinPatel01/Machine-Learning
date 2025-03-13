import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define categorical columns and encoders
cat_columns = ["job", "marital", "education", "contact", "month", "poutcome"]
categorical_encoders = {}

# Sample categories (replace with actual categories from dataset)
categories = {
    "job": ["admin", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"],
    "marital": ["married", "single", "divorced"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "poutcome": ["success", "failure", "other", "unknown"]
}

# Create label encoders for each categorical column
for col in cat_columns:
    encoder = LabelEncoder()
    encoder.fit(categories[col])
    categorical_encoders[col] = encoder

# Home Page
def home_page():
    st.title("Term Deposit Prediction App")
    st.image("https://miro.medium.com/v2/resize:fit:828/format:webp/1*HV3GT7juAbyeDwLt1Y621w.png", caption="Bank Term Deposit", use_container_width=True)
    st.write("## What is a Bank Term Deposit?")
    st.write(
        "A bank term deposit is a fixed-term investment where money is deposited in a bank for a predetermined period, earning a fixed interest rate."
    )
    st.write("This application predicts whether a customer will subscribe to a term deposit based on their details.")

    st.write("## About the Project")
    st.write("This project analyzes customer data to predict if they will subscribe to a term deposit. The prediction is made using machine learning techniques, specifically the XGBoost model, which is known for its high efficiency and accuracy in predictive modeling.")


 # Form Page
def form_page():
    st.title("Predict Term Deposit Subscription")
    
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", categories["job"])
    marital = st.selectbox("Marital Status", categories["marital"])
    education = st.selectbox("Education", categories["education"])
    balance = st.number_input("Balance", min_value=-10000, max_value=100000, value=1000)
    contact = st.selectbox("Contact Type", categories["contact"])
    month = st.selectbox("Last Contact Month", categories["month"])
    day = st.number_input("Last Contact Day", min_value=1, max_value=31, value=15)
    duration = st.number_input("Call Duration (seconds)", min_value=0, max_value=5000, value=100)
    campaign = st.number_input("Campaign Contacts", min_value=1, max_value=50, value=1)
    previous = st.number_input("Previous Contacts", min_value=0, max_value=50, value=0)
    poutcome = st.selectbox("Previous Outcome", categories["poutcome"])
    housing_new = st.selectbox("Has Housing Loan", [0, 1])
    loan_new = st.selectbox("Has Personal Loan", [0, 1])
    
    if st.button("Predict"):
        input_data = pd.DataFrame({
            "age": [age], "job": [job], "marital": [marital], "education": [education], "balance": [balance],
            "contact": [contact], "month": [month], "day": [day], "duration": [duration], "campaign": [campaign],
            "previous": [previous], "poutcome": [poutcome], "housing_new": [housing_new], "loan_new": [loan_new]
        })
        
        for col in cat_columns:
            input_data[col] = categorical_encoders[col].transform(input_data[col]).astype(int)  # Ensure numeric type


        
        # Ensure column order matches the training data
        input_data = input_data[model.feature_names_in_]
        prediction = model.predict(input_data)[0]

        result = "Subscribed" if prediction == 1 else "Not Subscribed"
        st.write(f"### Prediction: {result}")  

# Model Statistics Page
def model_statistics():
    st.title("Model Statistics")
    st.write("### Algorithm Used: XGBoost")
    st.write("XGBoost is a gradient boosting algorithm known for high performance and efficiency in predictive modeling.")
    st.write("### Model Accuracy: 89%")


    df = pd.read_csv("bank.csv")

    st.write("### Sample Data (20 rows)")
    st.write(df.sample(20))

    st.write("### Feature Importance")
    feature_importances = model.feature_importances_
    feature_names = model.feature_names_in_

    fig, ax = plt.subplots()
    sns.barplot(x=feature_importances, y=feature_names, ax=ax)
    ax.set_title("Feature Importance in Prediction")
    st.pyplot(fig)


# About Page
def about_page():
    st.title("About the Term Deposit Prediction Project")

    st.write("## Project Overview")
    st.write("""
    This project predicts whether a customer will subscribe to a term deposit using past marketing campaign data. 
    The prediction is based on customer details, call history, and past interactions with the bank. 
    The dataset used comes from a real-world banking campaign and contains various socio-economic and marketing-related attributes.
    """)

    st.write("## What is a Term Deposit?")
    st.write("""
    A **term deposit**, also known as a fixed deposit, is a type of financial product where a customer deposits a fixed sum of money into a bank for a predetermined period. 
    In return, the bank offers a guaranteed interest rate on the deposit. The customer cannot withdraw the money before the term ends without incurring a penalty.
    
    Banks often use **telemarketing campaigns** to encourage customers to subscribe to term deposits, and this project aims to predict the likelihood of a customer subscribing.
    """)

    st.write("## Key Features Used in Prediction")
    st.write("""
    The model is trained on several features from past banking campaigns. Below are the most important features:
    
    - **Age:** The age of the customer. Certain age groups are more likely to invest in term deposits.
    - **Job:** The type of job a customer has (e.g., management, technician, blue-collar). Some professions tend to be more financially stable.
    - **Marital Status:** Whether the customer is single, married, or divorced. This can influence financial decision-making.
    - **Education Level:** Higher education levels may indicate better financial literacy, affecting investment decisions.
    - **Balance:** The customer’s bank account balance. Customers with higher balances might be more likely to subscribe.
    - **Contact Type:** Whether the contact was made via cellular or telephone. Different contact methods can affect response rates.
    - **Month & Day of Contact:** The month and day when the customer was last contacted. Certain months may have better campaign success rates.
    - **Call Duration:** The duration of the last call with the customer. Longer calls often indicate more interest in the offer.
    - **Campaign Contacts:** The number of times the customer was contacted in the current campaign. Too many calls can result in negative responses.
    - **Previous Contacts:** The number of times the customer was contacted in past campaigns.
    - **Outcome of Previous Campaign:** Whether the customer subscribed in past campaigns. This can be a strong indicator of future behavior.
    - **Housing Loan & Personal Loan:** Whether the customer has a housing or personal loan. Customers with loans might have financial commitments that affect their decisions.
    """)

    st.write("## Libraries Used in This Project")
    st.write("""
    - **Streamlit:** A lightweight Python framework for building interactive web applications. Used to create the user interface for this project.
    - **Pandas:** A powerful data manipulation library that helps in loading, processing, and transforming data for machine learning.
    - **Matplotlib & Seaborn:** Libraries used for data visualization to analyze trends in customer behavior.
    - **Scikit-learn:** Provides tools for data preprocessing, encoding categorical variables, and evaluating the model.
    - **XGBoost:** A high-performance gradient boosting algorithm used for prediction. It optimizes decision trees and provides highly accurate results.
    - **Pickle:** Used for saving and loading the trained model to avoid re-training every time the application runs.
    """)

    st.write("## How Does This Project Work?")
    st.write("""
    1. **User Inputs:** Customers’ details are collected through the form in the app.
    2. **Data Preprocessing:** The input data is transformed and encoded to match the model’s training data.
    3. **Prediction:** The **XGBoost model** processes the input data and predicts whether the customer will subscribe to the term deposit.
    4. **Result Display:** The application displays the prediction in a user-friendly manner.
    """)

    st.write("## Why is This Project Useful?")
    st.write("""
    - **For Banks & Financial Institutions:** Helps target potential customers efficiently, reducing marketing costs.
    - **For Customers:** Ensures they receive personalized financial offers suited to their needs.
    - **For Data Science Enthusiasts:** Provides a real-world application of machine learning in finance.
    """)

    st.write("### Want to Try It Out?")
    st.write("Navigate to the **Form Page** to enter customer details and see the prediction!")



# Main Function
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Form", "Model Statistics", "About"])
    if page == "Home":
        home_page()
    elif page == "Form":
        form_page()
    elif page == "Model Statistics":
        model_statistics()
    elif page == "About":
        about_page()

if __name__ == "__main__":
    main()

