import streamlit as st
from prediction_helper import predict  # Ensure this is correctly linked to your prediction_helper.py

# Set the page configuration and title
st.set_page_config(page_title="VK Finance: Credit Risk Modelling", page_icon="📊")
st.title("VK Finance: Credit Risk Modelling")

# Create rows of three columns each
row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)
row5 = st.columns(3)

# Assign inputs to the first row with default values
with row1[0]:
    age = st.number_input('Age', min_value=18, step=1, max_value=100, value=28)
with row1[1]:
    loan_purpose = st.selectbox('Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'])
with row1[2]:
    loan_type = st.selectbox('Loan Type', ['Unsecured', 'Secured'])


with row2[0]:
    num_open_accounts = st.number_input('Open Loan Accounts', min_value=1, max_value=4, step=1, value=2)
with row2[1]:
    loan_tenure_months = st.number_input('Loan Tenure (months)', min_value=0, step=1, value=36)
with row2[2]:
    residence_type = st.selectbox('Residence Type', ['Owned', 'Rented', 'Mortgage'])

with row3[0]:
    income = st.number_input('Income', min_value=0, value=1200000)
with row3[1]:
    loan_amount = st.number_input('Loan Amount', min_value=0, value=2560000)
# Calculate Loan to Income Ratio and display it
loan_to_income_ratio = loan_amount / income if income > 0 else 0
with row3[2]:
    st.text("Loan to Income Ratio:")
    st.text(f"{loan_to_income_ratio:.2f}")  # Display as a text field

with row4[0]:
    delinquent_months = st.number_input('Deliquent Months', min_value=0, step=1, value=12)
with row4[1]:
    total_loan_months = st.number_input('Total Loan Month', min_value=0, step=1, value=36)
# calculate deliquency ratio
delinquency_ratio = round(
    (delinquent_months * 100 / total_loan_months) if total_loan_months != 0 else 0,
    2
)
with row4[2]:
    st.text("Deliquency Ratio:")
    st.text(f"{delinquency_ratio:.2f}")

with row5[0]:
    total_dpd = st.number_input('Total Due Pass Days', min_value=0, step=1, value=48)
#calculate avg dpd per deliquency
avg_dpd_per_delinquency =  round(
    total_dpd / delinquent_months if delinquent_months != 0 else 0,
    2
)
with row5[1]:
    st.text("DPD per deliqeuncy months:")
    st.text(f"{avg_dpd_per_delinquency:.2f}")
with row5[2]:
    credit_utilization_ratio = st.number_input('Credit Utilization Ratio', min_value=0, max_value=100, step=1, value=30)




# Button to calculate risk
if st.button('Calculate Risk'):
    # Call the predict function from the helper module
    # print((age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
    #                                             delinquency_ratio, credit_utilization_ratio, num_open_accounts,
    #                                             residence_type, loan_purpose, loan_type))
    probability, credit_score, rating = predict(age, loan_purpose, loan_type, num_open_accounts, loan_tenure_months,
                                                residence_type, income, loan_amount, loan_to_income_ratio, delinquent_months,
                                                total_loan_months, delinquency_ratio, total_dpd, avg_dpd_per_delinquency, credit_utilization_ratio)

    # Display the results
    st.write(f"Deafult Probability: {probability:.2%}")
    st.write(f"Credit Score: {credit_score}")
    st.write(f"Rating: {rating}")

# Footer
# st.markdown('_Project From Codebasics ML Course_')
