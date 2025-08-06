import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Page setup
st.set_page_config(page_title="Lifestyle & Mental Wellness Dashboard", layout="wide")
st.title("💻 Lifestyle & Mental Wellness Dashboard")
st.caption("By Ayush Ranjan | Internship Project @ Cognifyz Technologies")

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

# Correlation Heatmap
st.subheader("🔍 Correlation Heatmap")
corr = df.corr()
fig1, ax1 = plt.subplots(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax1)
st.pyplot(fig1)

# Histogram of screen time
st.subheader("⏰ Screen Time Distribution")
fig2, ax2 = plt.subplots()
df['daily_screen_time_hours'].hist(bins=15, color='skyblue', edgecolor='black', ax=ax2)
ax2.set_xlabel("Daily Screen Time (hrs)")
ax2.set_ylabel("Number of Users")
st.pyplot(fig2)

# Stress Risk Estimator (Regression Model)
st.subheader("🎯 Stress Level Predictor (Interactive)")

# Train model
X = df[['daily_screen_time_hours', 'caffeine_intake_mg_per_day']]
y = df['stress_level/10']
model = LinearRegression()
model.fit(X, y)

# Input sliders
col_input1, col_input2 = st.columns(2)
user_screen = col_input1.slider("Enter Your Daily Screen Time (hrs)", 0.0, 15.0, 5.0)
user_caffeine = col_input2.slider("Enter Your Daily Caffeine Intake (mg)", 0, 800, 200)

# Predict stress
predicted_stress = model.predict([[user_screen, user_caffeine]])[0]
st.success(f"🔮 Estimated Stress Level: **{predicted_stress:.2f}/10**")

# Footer
st.markdown("---")
st.markdown("<h6 style='text-align:center;color:gray;'>Created by Ayush Ranjan B.I.T MESRA | Powered by Streamlit</h6>", unsafe_allow_html=True)
