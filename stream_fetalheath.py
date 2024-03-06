import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv('fetal_health.csv')
    return df

# Train model
@st.cache
def train_model():
    df = load_data()
    X = df.drop('fetal_health', axis=1)
    y = df['fetal_health']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train, y_train)
    return rf_clf

# Sidebar
st.sidebar.title('Fetal Health Analysis')

# Sidebar options
option = st.sidebar.selectbox(
    'Choose an option:',
    ('Show raw data', 'Train Model', 'Predict')
)

# Show raw data
if option == 'Show raw data':
    st.subheader('Raw data')
    df = load_data()
    st.write(df)

# Train model
elif option == 'Train Model':
    st.subheader('Train Model')
    rf_clf = train_model()
    st.write('Model trained successfully!')

# Predict
elif option == 'Predict':
    st.subheader('Predict')

    # Input features
    input_features = {}
    input_features['baseline value'] = st.number_input('Baseline Value', value=120)
    input_features['accelerations'] = st.number_input('Accelerations', value=0.0)
    input_features['fetal_movement'] = st.number_input('Fetal Movement', value=0.0)
    input_features['uterine_contractions'] = st.number_input('Uterine Contractions', value=0.0)
    input_features['light_decelerations'] = st.number_input('Light Decelerations', value=0.0)
    input_features['severe_decelerations'] = st.number_input('Severe Decelerations', value=0.0)
    input_features['prolongued_decelerations'] = st.number_input('Prolongued Decelerations', value=0.0)
    input_features['abnormal_short_term_variability'] = st.number_input('Abnormal Short Term Variability', value=0.0)
    input_features['mean_value_of_short_term_variability'] = st.number_input('Mean Value of Short Term Variability', value=0.0)
    input_features['percentage_of_time_with_abnormal_long_term_variability'] = st.number_input('Percentage of Time with Abnormal Long Term Variability', value=0.0)
    input_features['mean_value_of_long_term_variability'] = st.number_input('Mean Value of Long Term Variability', value=0.0)
    input_features['histogram_width'] = st.number_input('Histogram Width', value=0.0)
    input_features['histogram_min'] = st.number_input('Histogram Min', value=0.0)
    input_features['histogram_max'] = st.number_input('Histogram Max', value=0.0)
    input_features['histogram_number_of_peaks'] = st.number_input('Histogram Number of Peaks', value=0.0)
    input_features['histogram_number_of_zeroes'] = st.number_input('Histogram Number of Zeroes', value=0.0)
    input_features['histogram_mode'] = st.number_input('Histogram Mode', value=0.0)
    input_features['histogram_mean'] = st.number_input('Histogram Mean', value=0.0)
    input_features['histogram_median'] = st.number_input('Histogram Median', value=0.0)
    input_features['histogram_variance'] = st.number_input('Histogram Variance', value=0.0)
    input_features['histogram_tendency'] = st.number_input('Histogram Tendency', value=0.0)

    # Make prediction
    rf_clf = train_model()
    input_data = pd.DataFrame([input_features])
    prediction = rf_clf.predict(input_data)

    # Show prediction
    st.write('Predicted Fetal Health:', prediction[0])
