import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Handle missing values
    df = df.dropna()
    
    # Standardize numerical features
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['number']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

def fit_algorithm(df, target_col):
    # Split the dataset
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    
    return model, score

# Streamlit UI
st.set_page_config(page_title='Data Visualization App', layout='wide')
st.title('📊 Data Visualization with Preprocessing & Model Fitting')

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Dataset Preview")
    st.dataframe(df.head())
    
    # Preprocess Data
    df = preprocess_data(df)
    st.write("### Preprocessed Dataset Preview")
    st.dataframe(df.head())
    
    # Choose target variable
    target_col = st.selectbox("Select Target Column", df.columns)
    
    if target_col:
        model, score = fit_algorithm(df, target_col)
        st.write(f"### Model Performance: R² Score = {score:.4f}")
    
    # Visualization
    st.write("## Data Visualizations")
    
    col1, col2 = st.columns(2)
    
    # Seaborn Heatmap
    with col1:
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    
    # Plotly Scatter Plot
    with col2:
        st.write("### Scatter Plot")
        x_col = st.selectbox("Select X-axis", df.columns)
        y_col = st.selectbox("Select Y-axis", df.columns)
        fig = px.scatter(df, x=x_col, y=y_col, color=df.columns[0])
        st.plotly_chart(fig)
    
    # Matplotlib Histogram
    st.write("### Histogram of Numerical Features")
    fig, ax = plt.subplots(figsize=(10, 5))
    df.hist(ax=ax, bins=20, color='skyblue', edgecolor='black')
    st.pyplot(fig)
