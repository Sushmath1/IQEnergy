from main import train_and_predict
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib 
import pandas as pd
import os
st.set_page_config(layout="wide")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "demand_model.pkl")

demand_model = joblib.load(model_path)

past_demand, past_renewable, future_demand, future_renewable = train_and_predict()

avg_demand = int(np.mean(future_demand))
avg_renewable = int(np.mean(future_renewable))

surplus_deficit = avg_renewable - avg_demand

house_demand = int(avg_demand * 0.6)
industry_demand = int(avg_demand * 0.4)

left_col, right_col = st.columns([0.8, 2.2])

with left_col:
    st.markdown("### 🤖 Ask Any Related Queries!")

    user_question = st.text_input("Ask about smart grids:")

    def chatbot_response(q):
        q = q.lower()

        if any(word in q for word in ["hi", "hello", "hey"]):
            return "Hello!"

        elif "work" in q:
            return (
                "This system trains ML models on historical data, "
                "predicts demand & renewable output for the next 4 hours, "
                "and detects surplus or deficit."
            )

        elif "surplus" in q:
            return "Surplus = Renewable generation is greater than demand."

        elif "deficit" in q:
            return "Deficit = Demand is greater than renewable generation."

        elif "renewable" in q:
            return "Renewable energy comes from solar and wind sources."

        else:
            return "Try asking: How does this system work?"

    if user_question:
        st.info(chatbot_response(user_question))

with right_col:

    st.title("⚡ EnergyIQ Dashboard")
    st.subheader("Smart Grid – ML-Based 4 Hour Forecast")
    st.divider()

    colA, colB = st.columns(2)

    with colA:
        st.markdown("## 🔍 Grid Status")

        if surplus_deficit >= 0:
            st.success(f"🟢 SURPLUS: {surplus_deficit} MW Available")
        else:
            st.error(f"🔴 DEFICIT: {abs(surplus_deficit)} MW Shortage")

    with colB:
        st.markdown("## 🏠 Distributed Demand")
        st.write(f"Residential: {house_demand} MW")
        st.write(f"Industrial: {industry_demand} MW")
        st.write(f"Total: {house_demand + industry_demand} MW")

    st.divider()

    st.markdown("### ⏳ Exact 4 Hour Prediction")

    if st.button("Show 4 Hour Forecast"):
        for i in range(4):
            st.write(
                f"Hour {i+1}: "
                f"Demand = {int(future_demand[i])} MW | "
                f"Renewable = {int(future_renewable[i])} MW"
            )

    st.divider()

    st.markdown("### 🧭 Operational Guidelines")

    if surplus_deficit >= 0:
        st.info("""
        • Reduce power generation slightly (around 2–5%) to avoid overloading the grid.  
        • Store extra electricity in batteries if available.  
        • Supply excess power to nearby regions if possible.  
        • Use this time to charge electric vehicles or run heavy loads. 
        """)
    else:
        st.warning("""
        • Increase power generation by 3–7% if backup sources are available.  
        • Activate thermal or backup generators if needed.  
        • Reduce non-essential industrial loads temporarily.  
        • Encourage consumers to reduce heavy appliance usage 
        """)

    st.divider()

    st.markdown("## 📊 Forecast Graphs")

    graph_option = st.selectbox(
        "Select Graph Type",
        (
            "Energy Demand Prediction",
            "Renewable Energy Prediction"
        )
    )

    plt.figure()

    if graph_option == "Energy Demand Prediction":
        plt.plot(past_demand, label="Past Demand")
        plt.plot(
            range(len(past_demand), len(past_demand) + len(future_demand)),
            future_demand,
            linestyle="dotted",
            color="red",
            label="Future Demand (Next 4 hrs)"
        )

    elif graph_option == "Renewable Energy Prediction":
        plt.plot(past_renewable, label="Past Renewable")
        plt.plot(
            range(len(past_renewable), len(past_renewable) + len(future_renewable)),
            future_renewable,
            linestyle="dotted",
            color="green",
            label="Future Renewable (Next 4 hrs)"
        )

    plt.xlabel("Time (Hours)")
    plt.ylabel("Power (MW)")
    plt.legend()
    st.pyplot(plt)

st.caption("EnergyIQ – ML-Based Smart Grid Forecasting System (Educational Project)")

print(model_path)
print(os.path.exists(model_path))












