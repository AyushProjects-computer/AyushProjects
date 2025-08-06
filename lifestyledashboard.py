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

st.subheader("📉 Screen Time vs Stress Level")

fig4, ax4 = plt.subplots()
sns.regplot(x='daily_screen_time_hours', y='stress_level/10', data=df, ax=ax4, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
ax4.set_title("Relation Between Screen Time and Stress Level")
ax4.set_xlabel("Daily Screen Time (hrs)")
ax4.set_ylabel("Stress Level (/10)")
st.pyplot(fig4)

st.subheader("☕ Average Caffeine Intake by Gender")

avg_caffeine_gender = df.groupby('gender')['caffeine_intake_mg_per_day'].mean().reset_index()

fig5, ax5 = plt.subplots()
sns.barplot(x='gender', y='caffeine_intake_mg_per_day', data=avg_caffeine_gender, palette='Set2', ax=ax5)
ax5.set_ylabel("Avg Caffeine (mg/day)")
ax5.set_title("Caffeine Intake by Gender")
st.pyplot(fig5)
st.subheader("🧠 Stress Level Distribution")

stress_bins = pd.cut(df['stress_level/10'], bins=[0, 3, 6, 10], labels=["Low", "Moderate", "High"])
stress_counts = stress_bins.value_counts().sort_index()

fig6, ax6 = plt.subplots()
ax6.pie(stress_counts, labels=stress_counts.index, autopct='%1.1f%%', colors=['#8BC34A', '#FFC107', '#F44336'])
ax6.set_title("Distribution of Stress Levels")
st.pyplot(fig6)


# Footer
st.markdown("---")
st.markdown("<h6 style='text-align:center;color:gray;'>Created by Ayush Ranjan B.I.T MESRA | Powered by Streamlit</h6>", unsafe_allow_html=True)
st.markdown("""
---
📌 **Health Tip of the Day**  
*Drink water, take digital detox breaks, and limit caffeine to improve your mental clarity.*
""")
