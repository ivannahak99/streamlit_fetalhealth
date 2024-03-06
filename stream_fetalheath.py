import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv('fetal_health.csv')
    return df

# Sidebar
st.sidebar.title('Fetal Health Analysis')

# Load data
df = load_data()

# Sidebar options
option = st.sidebar.selectbox(
    'Choose an option:',
    ('Show raw data', 'Class Distribution', 'Feature Correlation', 'Feature Distribution')
)

# Show raw data
if option == 'Show raw data':
    st.subheader('Raw data')
    st.write(df)

# Class distribution
elif option == 'Class Distribution':
    st.subheader('Class Distribution')
    plt.figure(figsize=(8, 5))
    sns.countplot(x='fetal_health', data=df)
    plt.title('Distribution of Fetal Health Classes')
    plt.xlabel('Fetal Health')
    plt.ylabel('Count')
    st.pyplot()

# Feature correlation
elif option == 'Feature Correlation':
    st.subheader('Feature Correlation')
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Features')
    st.pyplot()

# Feature distribution
elif option == 'Feature Distribution':
    st.subheader('Feature Distribution')
    for col in df.columns[:-1]:  # Exclude the target column
        plt.figure(figsize=(8, 5))
        sns.histplot(x=col, hue="fetal_health", data=df, kde=True)
        plt.title(f'Distribution of {col} by Fetal Health')
        plt.xlabel(col)
        plt.ylabel('Count')
        st.pyplot()
