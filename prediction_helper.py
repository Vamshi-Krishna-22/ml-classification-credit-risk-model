import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load model and components
MODEL_PATH = 'artifacts/model_data.joblib'
model_data = joblib.load(MODEL_PATH)

model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
cols_to_scale = model_data['cols_to_scale']


def prepare_input(age, loan_purpose, loan_type, num_open_accounts, loan_tenure_months,
                  residence_type, income, loan_amount, loan_to_income_ratio,
                  delinquent_months, total_loan_months, delinquency_ratio,
                  total_dpd, avg_dpd_per_deliquency, credit_utilization_ratio):
    input_data = {
        'age': age,
        'income': income,
        'loan_amount': loan_amount,
        'loan_to_income': loan_to_income_ratio,
        'loan_tenure_months': loan_tenure_months,
        'total_loan_months': total_loan_months,
        'delinquent_months': delinquent_months,
        'total_dpd': total_dpd,
        'avg_dpd_per_deliquency': avg_dpd_per_deliquency,
        'delinquency_ratio': delinquency_ratio,
        'credit_utilization_ratio': credit_utilization_ratio,
        'number_of_open_accounts': num_open_accounts,

        # One-hot encoding for residence_type
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,

        # One-hot encoding for loan_purpose
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,

        # One-hot encoding for loan_type
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,

        # Dummy fields (as expected by scaler and model)
        'number_of_dependants': 1,
        'years_at_current_address': 1,
        'zipcode': 1,
        'sanction_amount': 1,
        'processing_fee': 1,
        'gst': 1,
        'net_disbursement': 1,
        'principal_outstanding': 1,
        'bank_balance_at_application': 1,
        'number_of_closed_accounts': 1,
        'enquiry_count': 1
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Scale only selected columns
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Ensure final DataFrame matches the model’s expected feature list
    df = df[features]

    return df


def predict(age, loan_purpose, loan_type, num_open_accounts, loan_tenure_months,
            residence_type, income, loan_amount, loan_to_income_ratio,
            delinquent_months, total_loan_months, delinquency_ratio,
            total_dpd, avg_dpd_per_deliquency, credit_utilization_ratio):
    input_df = prepare_input(
        age, loan_purpose, loan_type, num_open_accounts, loan_tenure_months,
        residence_type, income, loan_amount, loan_to_income_ratio,
        delinquent_months, total_loan_months, delinquency_ratio,
        total_dpd, avg_dpd_per_deliquency, credit_utilization_ratio
    )

    return calculate_credit_score(input_df)


def calculate_credit_score(input_df, base_score=300, scale_length=600):
    x = np.dot(input_df.values, model.coef_.T) + model.intercept_

    default_probability = 1 / (1 + np.exp(-x))
    non_default_probability = 1 - default_probability
    credit_score = base_score + non_default_probability.flatten() * scale_length

    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'

    rating = get_rating(credit_score[0])

    return default_probability.flatten()[0], int(credit_score[0]), rating
