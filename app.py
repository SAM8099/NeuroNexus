import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from PIL import Image

# Import the prediction classes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Set page config
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# Title and description
st.title("🚢 Titanic Survival Prediction")
st.markdown("""
This application predicts whether a passenger would have survived the Titanic disaster based on their details.
""")

# Create two columns for the form
col1, col2 = st.columns(2)

# Input form
with st.form("prediction_form"):
    st.subheader("Enter Passenger Details")
    
    # Column 1
    with col1:
        pclass = st.selectbox(
            "Passenger Class",
            options=[1, 2, 3],
            help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class"
        )
        
        sex = st.selectbox(
            "Gender",
            options=["male", "female"]
        )
        
        age = st.number_input(
            "Age",
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=1.0
        )
        
        sibsp = st.number_input(
            "Number of Siblings/Spouses Aboard",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )
    
    # Column 2
    with col2:
        parch = st.number_input(
            "Number of Parents/Children Aboard",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )
        
        fare = st.number_input(
            "Fare (in pounds)",
            min_value=0.0,
            max_value=600.0,
            value=30.0,
            step=10.0
        )
        
        embarked = st.selectbox(
            "Port of Embarkation",
            options=["C", "Q", "S"],
            help="C = Cherbourg, Q = Queenstown, S = Southampton"
        )
    
    submit_button = st.form_submit_button("Predict Survival")

# When the form is submitted
if submit_button:
    # Display loading message
    with st.spinner("Making prediction..."):
        # Create data instance
        data = CustomData(
            pclass=pclass,
            sex=sex,
            age=age,
            sibsp=sibsp,
            parch=parch,
            fare=fare,
            embarked=embarked
        )
        
        # Get DataFrame
        df = data.get_data_as_dataframe()
        
        # Show the input data
        st.subheader("Input Data")
        st.dataframe(df)
        
        # Initialize prediction pipeline
        predict_pipeline = PredictPipeline()
        
        try:
            # Make prediction
            results = predict_pipeline.predict(df)
            
            # Display results
            st.subheader("Prediction Result")
            
            if results[0] == 1:
                st.success("This passenger would likely **SURVIVE** ✅")
            else:
                st.error("This passenger would likely **NOT SURVIVE** ❌")
            
            # Additional information about the prediction
            survival_chances = "High" if results[0] == 1 else "Low"
            
            st.markdown(f"""
            ### Passenger Profile:
            - **Class:** {'1st' if pclass == 1 else '2nd' if pclass == 2 else '3rd'} Class
            - **Gender:** {sex.capitalize()}
            - **Age:** {age}
            - **Family Members Aboard:** {sibsp + parch}
            - **Embarkation Port:** {'Cherbourg' if embarked == 'C' else 'Queenstown' if embarked == 'Q' else 'Southampton'}
            - **Survival Chances:** {survival_chances}
            """)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Add historical context and information
with st.expander("About the Titanic Disaster"):
    st.markdown("""
    ### The Titanic Disaster
    
    The RMS Titanic sank on April 15, 1912, after colliding with an iceberg during her maiden voyage from Southampton to New York City. 
    Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making it one of the deadliest commercial maritime disasters in modern history.
    
    ### Survival Factors
    
    Several factors influenced survival rates:
    - **Women and children first:** The "women and children first" protocol was generally followed during evacuation.
    - **Class:** First-class passengers had better access to lifeboats.
    - **Age:** Children were more likely to be given priority.
    - **Family size:** Traveling alone or with a small family might have affected survival chances.
    """)

# Add information about the model
with st.expander("About the Model"):
    st.markdown("""
    ### Machine Learning Model
    
    This prediction is based on a machine learning model trained on historical data from the Titanic disaster. 
    The model considers various factors like passenger class, gender, age, and other features to determine the likelihood of survival.
    
    The model was trained using scikit-learn and performs with reasonable accuracy on test data.
    
    ### Disclaimer
    
    This is a predictive model and should be used for educational purposes only. The actual events of the Titanic disaster were complex and many individual stories of survival or loss were influenced by factors not captured in this model.
    """)

# Footer
st.markdown("---")
st.markdown("### Created for NeuroNexus Internship Project")