# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st

st.header("A web-based application for predicting persistent critical illness among ICU orthopaedic trauma patients using ensemble machine learning")
st.sidebar.title("Selection of Parameters")
st.sidebar.markdown("Picking up parameters")

Pelvic_fractrue = st.sidebar.selectbox("Pelvic fractrue", ("No", "Closed", "Open"))
Ventilation = st.sidebar.selectbox("Mechanical ventilation", ("No", "Yes"))
Respiratory_failure = st.sidebar.selectbox("Respiratory failure", ("No", "Yes"))
Pneumonia = st.sidebar.selectbox("Pneumonia", ("No", "Yes"))
Sepsis = st.sidebar.selectbox("Sepsis", ("No", "Yes"))
Heart_rate = st.sidebar.slider("Heart rate (Beats per minute)", 45, 135)
Respiratory_rate = st.sidebar.slider("Respiratory rate (Beats per minute)", 10, 40)
Albumin = st.sidebar.slider("Albumin (g/dL)", 1.00, 6.00)
Glucose = st.sidebar.slider("Glucose (mg/dL)", 90.0, 190.0)
Calcium = st.sidebar.slider("Total serum calcium (mg/dL)", 3.00, 13.00)
Hematocrit = st.sidebar.slider("Hematocrit (%)", 25.00, 45.00)
OASIS = st.sidebar.slider("OASIS",  10, 60)
SAPSII = st.sidebar.slider("SAPSII", 10, 60)
SOFA = st.sidebar.slider("SOFA", 0, 20)

if st.button("Submit"):
    Xgbc_clf = jl.load("Xgbc_clf_final_round.pkl")
    x = pd.DataFrame([[Pelvic_fractrue, Ventilation, Respiratory_failure, Pneumonia, Sepsis, Heart_rate, Respiratory_rate, Albumin, Glucose, Calcium, Hematocrit, OASIS,
                              SAPSII, SOFA]],
                     columns=["Pelvicfractrue", "Ventilation", "Respiratoryfailure", "Pneumonia", "Sepsis", "Heartrate", "Respiratoryrate", "Albumin", "Glucose", "Calcium", "Hematocrit", "OASIS",
                              "SAPSII", "SOFA"])

    x = x.replace(["No", "Yes"], [0, 1])
    x = x.replace(["No", "Closed", "Open"], [0, 1, 2])

    # Get prediction
    prediction = Xgbc_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.text(f"Probability of developing PerCI: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.100:
        st.text(f"Risk group: low-risk group")
    else:
        st.text(f"Risk group: High-risk group")

st.subheader('Introduction')
st.markdown('The model was developed based on the ensembling technique with the Baier score of 0.097 and AUC of 0.814 (95% CI: 0.739-0.889). According to the best cut-off value, patients with a predicted risk of less than 0.100 were categorized into the low-risk group, while patients with a predicted risk of more than 0.100 were categorized into the high-risk group.')
