import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Page setup
st.set_page_config(page_title="Lifestyle & Mental Wellness Dashboard", layout="wide")
st.title("💻 Lifestyle & Mental Wellness Dashboard")
st.caption("By Ayush Ranjan | Summer Training @ Cognifyz Technologies")

# Load dataset
df = pd.read_csv("digital_diet_mental_health.csv")

# Clean & Prepare
df.dropna(inplace=True)  # ensure no NaNs for model and plots

# Overview Metrics
st.header("📊 Overview Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("📱 Avg Screen Time", f"{df['daily_screen_time_hours'].mean():.2f} hrs")
col2.metric("☕ Avg Caffeine Intake", f"{df['caffeine_intake_mg_per_day'].mean():.0f} mg")
col3.metric("😰 Avg Stress Level", f"{df['stress_level/10'].mean():.2f}/10")

# Gender Analysis
st.subheader("👩‍⚕️ Gender-Based Averages")
gender_df = df.groupby('gender')[['daily_screen_time_hours', 'caffeine_intake_mg_per_day']].mean()
st.bar_chart(gender_df)

st.subheader("🔍 Pairwise Relationship Overview")

numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
sample_df = df[numerical_cols].dropna().sample(n=100)  # Small sample for performance

fig3 = sns.pairplot(sample_df)
st.pyplot(fig3)


# Footer
st.markdown("---")
st.markdown("<h6 style='text-align:center;color:gray;'>Created by Ayush Ranjan B.I.T MESRA | Powered by Streamlit</h6>", unsafe_allow_html=True)
