import streamlit as st
from sklearn.datasets import fetch_openml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cache the data loading function for faster processing
@st.cache_data
def load_data():
    titanic_sklearn = fetch_openml('titanic', version=1, as_frame=True)
    titanic_df = titanic_sklearn.frame
    return titanic_df

# Load data
data = load_data()

# Set the title of the app
st.title("Titanic Data Analysis")

# Sidebar for user inputs
st.sidebar.header("Filter Options")

# User Inputs
# 1. Select Passenger Class
pclass = st.sidebar.multiselect(
    'Passenger Class', options=[1, 2, 3], default=[1, 2, 3]
)

# 2. Select Gender
gender = st.sidebar.multiselect(
    'Gender', options=['male', 'female'], default=['male', 'female']
)

# 3. Select Age Range
age_min, age_max = st.sidebar.slider(
    'Select Age Range', min_value=0, max_value=80, value=(0, 80)
)

# 4. Select Fare Range
fare_min, fare_max = st.sidebar.slider(
    'Select Fare Range', min_value=0, max_value=int(data['fare'].max()), value=(0, int(data['fare'].max()))
)

# Filter data based on user input
filtered_data = data[
    (data['pclass'].astype(int).isin(pclass)) &
    (data['sex'].isin(gender)) &
    (data['age'].astype(float) >= age_min) &
    (data['age'].astype(float) <= age_max) &
    (data['fare'].astype(float) >= fare_min) &
    (data['fare'].astype(float) <= fare_max)
]

# Tabs for layout
tab1, tab2 = st.tabs(["Data Overview", "Visualization"])

with tab1:
    st.header("Filtered Data")
    st.dataframe(filtered_data)

    # Summary statistic
    st.subheader("Summary Statistics")
    if st.checkbox("Show Summary Statistics"):
        st.write(filtered_data.describe())

with tab2:
    st.header("Visualization")
    # User can select the type of plot
    plot_type = st.selectbox(
        "Select Plot Type",
        options=["Survival Rate by Gender", "Age Distribution"]
    )

    if plot_type == "Survival Rate by Gender":
        # Plot Survival Rate by Gender
        sns.set_style('whitegrid')
        fig, ax = plt.subplots()
        sns.countplot(data=filtered_data, x='sex', hue='survived', ax=ax)
        ax.set_title('Survival Rate by Gender')
        ax.set_xlabel('Gender')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    elif plot_type == "Age Distribution":
        # Plot Age Distribution
        fig, ax = plt.subplots()
        sns.histplot(filtered_data['age'].astype(float), bins=30, kde=True, ax=ax)
        ax.set_title('Age Distribution')
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
