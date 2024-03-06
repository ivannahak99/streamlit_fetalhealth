import streamlit as st
import pandas as pd

# Membaca file CSV
@st.cache
def load_data():
    data = pd.read_csv("fetal_health.csv")
    return data

# Menampilkan data
def main():
    st.title("Data Fetal Health")
    data = load_data()
    st.write(data)

if __name__ == "__main__":
    main()
