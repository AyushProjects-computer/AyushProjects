import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------#
# Page Setup
# -----------------------------#
st.set_page_config(page_title="Lifestyle & Mental Wellness Dashboard", layout="wide")
st.title("💻 Lifestyle & Mental Wellness Dashboard")
st.caption("By Ayush Ranjan | Summer Training @ Cognifyz Technologies")

# -----------------------------#
# Load Data
# -----------------------------#
df = pd.read_csv("digital_diet_mental_health.csv")
df.dropna(inplace=True)

# -----------------------------#
# Overview Section
# -----------------------------#
st.header("📊 Overview Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("📱 Avg Screen Time", f"{df['daily_screen_time_hours'].mean():.2f} hrs")
col2.metric("☕ Avg Caffeine Intake", f"{df['caffeine_intake_mg_per_day'].mean():.0f} mg")
col3.metric("😰 Avg Stress Level", f"{df['stress_level/10'].mean():.2f}/10")

# -----------------------------#
# Gender-Based Screen & Caffeine Bar Chart
# -----------------------------#
st.subheader("👩‍⚕️ Gender-Based Averages")
gender_df = df.groupby('gender')[['daily_screen_time_hours', 'caffeine_intake_mg_per_day']].mean()
st.bar_chart(gender_df)

# -----------------------------#
# Visual 1: Screen Time vs Stress Level
# -----------------------------#
st.subheader("📉 Screen Time vs Stress Level")
fig4, ax4 = plt.subplots()
sns.regplot(x='daily_screen_time_hours', y='stress_level/10', data=df, ax=ax4,
            scatter_kws={"color": "blue"}, line_kws={"color": "red"})
ax4.set_title("Relation Between Screen Time and Stress Level")
ax4.set_xlabel("Daily Screen Time (hrs)")
ax4.set_ylabel("Stress Level (/10)")
st.pyplot(fig4)

# -----------------------------#
# Visual 2: Avg Caffeine Intake by Gender
# -----------------------------#
st.subheader("☕ Average Caffeine Intake by Gender")
avg_caffeine_gender = df.groupby('gender')['caffeine_intake_mg_per_day'].mean().reset_index()
fig5, ax5 = plt.subplots()
sns.barplot(x='gender', y='caffeine_intake_mg_per_day', data=avg_caffeine_gender, palette='Set2', ax=ax5)
ax5.set_ylabel("Avg Caffeine (mg/day)")
ax5.set_title("Caffeine Intake by Gender")
st.pyplot(fig5)

# -----------------------------#
# Visual 3: Pie Chart - Stress Level Categories
# -----------------------------#
st.subheader("🧠 Stress Level Distribution")
stress_bins = pd.cut(df['stress_level/10'], bins=[0, 3, 6, 10], labels=["Low", "Moderate", "High"])
stress_counts = stress_bins.value_counts().sort_index()
fig6, ax6 = plt.subplots()
ax6.pie(stress_counts, labels=stress_counts.index, autopct='%1.1f%%',
        colors=['#8BC34A', '#FFC107', '#F44336'])
ax6.set_title("Distribution of Stress Levels")
st.pyplot(fig6)

# -----------------------------#
# Live Prediction: Stress Estimator
# -----------------------------#
st.subheader("🎯 Stress Level Estimator (Interactive)")
X = df[['daily_screen_time_hours', 'caffeine_intake_mg_per_day']]
y = df['stress_level/10']
model = LinearRegression()
model.fit(X, y)

col_input1, col_input2 = st.columns(2)
user_screen = col_input1.slider("📱 Your Daily Screen Time (hrs)", 0.0, 15.0, 5.0)
user_caffeine = col_input2.slider("☕ Your Daily Caffeine Intake (mg)", 0, 800, 200)

predicted_stress = model.predict([[user_screen, user_caffeine]])[0]
st.success(f"🔮 Estimated Stress Level: **{predicted_stress:.2f}/10**")

# Feedback based on inputs
st.subheader("🧠 Lifestyle Feedback")
if user_screen > 8:
    st.warning("⚠️ High screen time detected. Consider taking frequent breaks.")
elif user_screen > 5:
    st.info("🙂 Moderate screen time. Stay aware.")
else:
    st.success("✅ Good screen management!")

if user_caffeine > 400:
    st.warning("☕ High caffeine consumption. Reduce for better sleep and clarity.")
elif user_caffeine > 200:
    st.info("⚠️ Moderate caffeine intake. Monitor for jitters or sleep impact.")
else:
    st.success("👍 Caffeine level within healthy range.")

# -----------------------------#
# Footer
# -----------------------------#
st.markdown("---")
st.markdown("<h6 style='text-align:center;color:gray;'>Created by Ayush Ranjan | B.I.T MESRA | Powered by Streamlit</h6>",
            unsafe_allow_html=True)

st.markdown("""
---
📌 **Health Tip of the Day**  
*Take breaks from screens every 30–60 minutes. Drink water and limit caffeine intake for better focus.*
""")

