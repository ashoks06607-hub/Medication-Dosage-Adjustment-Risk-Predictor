import streamlit as st
import pandas as pd
import pickle

# Load Model

model = pickle.load(open("final_model_xg.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))


# App Title

st.title("💊 Medication Dosage Risk Prediction System")
st.write("Clinical Decision Support Tool (XGBoost Model)")

# User Input

def user_input():

    data = {}

    diabetesMed = st.selectbox("Patient currently on diabetes medication?", ["Yes", "No"])

    num_medications = st.number_input("Total number of medications prescribed", 0, 50, 5)

    A1Cresult = st.selectbox("Glycated haemoglobin (A1C) result", ["Norm", ">7", ">8"])

    procedures = st.selectbox("Any clinical procedures during admission?", ["No", "Yes"])

    time_in_hospital = st.number_input("Length of hospital stay (days)", 0, 14, 3)

    age = st.slider("Patient Age", 0, 100, 50)

    number_diagnoses = st.slider("Number of diagnosed conditions", 1, 20, 3)

    # Feature Engineering

    data["diabetesMed"] = 1 if diabetesMed == "Yes" else 0

    data["num_medications"] = num_medications

    data["med_complexity"] = 1 if num_medications >= 20 else 0

    A1C_map = {"Norm": 1, ">7": 2, ">8": 3}
    data["A1Cresult"] = A1C_map[A1Cresult]

    data["procedures"] = 1 if procedures == "Yes" else 0

    data["time_in_hospital"] = time_in_hospital

    data["age"] = age
    
    data["number_diagnoses"] = number_diagnoses

    return pd.DataFrame([data])


input_df = user_input()

# Prediction

if st.button("Predict Risk"):

    input_df = input_df.reindex(columns=columns, fill_value=0)

    probs = model.predict_proba(input_df)[0]
    
    low_risk_prob = probs[0]
    
    high_risk_prob = probs[1]

    prediction = model.predict(input_df)[0]

    risk_label = "HIGH RISK 🚨" if prediction == 1 else "LOW RISK ✅"

    # Output

    st.subheader("Prediction Result")
    st.write(f"**Risk Classification:** {risk_label}")
    
    st.write(f"**Probability of High Risk:** {round(high_risk_prob * 100, 2)}%")
    
    st.write(f"**Low Risk Probability:** {round(low_risk_prob * 100, 2)}%")

    # Clinical Explanation

    st.subheader("Clinical Decision Support Explanation")

    explanation = []

    if input_df["diabetesMed"].values[0] == 1:
        explanation.append("Patient is on diabetes medication, increasing sensitivity to dosage changes.")

    if input_df["med_complexity"].values[0] == 1:
        explanation.append("High medication burden (20+ medications) indicates increased treatment complexity.")

    if 10 < input_df["num_medications"].values[0] < 20:
        explanation.append("Moderate number of medications - potential interaction risk.")

    if input_df["A1Cresult"].values[0] in [2, 3]:
        explanation.append("Elevated A1C indicates suboptimal glycemic control.")

    if input_df["procedures"].values[0] == 1:
        explanation.append("History of procedures suggests higher clinical intervention complexity.")

    if input_df["time_in_hospital"].values[0] > 5:
        explanation.append("Extended hospital stay may indicate higher severity of condition.")

    if input_df["age"].values[0] > 65:
        explanation.append("Older age is associated with increased clinical risk.")

    if input_df["number_diagnoses"].values[0] > 5:
        explanation.append("Multiple diagnosed conditions indicate higher comorbidity burden.")
        
    if prediction == 0 and len(explanation) >= 2:
        explanation.append("⚠️ Multiple moderate risk factors are present, but overall risk remains low based on model assessment.")

    if len(explanation) == 0:
        explanation.append("No major high-risk clinical indicators detected.")

    for e in explanation:
        st.write("• " + e)

    st.markdown("---")
    st.info("This tool is for clinical decision support only and does not replace medical judgment.")