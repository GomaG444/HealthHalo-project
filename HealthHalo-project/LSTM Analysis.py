import streamlit as st
import random
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="HealthHalo - LSTM Weekly Insights", layout="centered")

# Title
st.title("HealthHalo LSTM Analysis")
st.subheader("Weekly Heart Health Insights")

# Possible weekly messages
weekly_alerts = [
    "Stable readings this week — no major anomalies.",
    "Irregular heart rate patterns detected 3× in the past week.",
    "Slight increase in blood pressure and cholesterol levels.",
    "Heart rhythm and vitals appear normal and consistent.",
    "Elevated heart disease risk detected over multiple days!",
    "Noticed reduced activity this week — keep moving!",
    "Excellent consistency in heart health data this week.",
]

# Generate weekly summaries for the last 4 weeks
def generate_weekly_alerts():
    today = datetime.today()
    summaries = []
    for i in range(4):
        week_end = today - timedelta(days=i * 7)
        week_start = week_end - timedelta(days=6)
        summaries.append({
            "week_range": f"{week_start.strftime('%b %d')} - {week_end.strftime('%b %d')}",
            "alert": random.choice(weekly_alerts)
        })
    return summaries[::-1]  # Show oldest to newest

# Display weekly summaries
weekly_summaries = generate_weekly_alerts()
st.markdown("### Past 4 Weeks Overview")

for summary in weekly_summaries:
    st.markdown(f"** {summary['week_range']}**")
    st.info(summary['alert'])

# Footer
st.markdown("---")
st.caption(" HealthHalo · AI-Powered Insights · Prototype v1.0")
