
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Gym Member Analytics",
    page_icon="🏋️",
    layout="wide"
)

model = joblib.load("gym_churn_model.pkl")
df = pd.read_csv("gym_member_analysis.csv")

st.title("🏋️ Gym Member Analytics & Churn Prediction")
st.write("Gym membership analysis and ML-based member status classification.")

# Gym Overview
st.subheader("Gym Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Members", len(df))
col2.metric("Active Members", int(df["is_active"].sum()))
col3.metric("Churned Members", int((~df["is_active"]).sum()))

st.divider()

# Membership Analysis
st.subheader("Membership Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("Active vs Inactive Members")

    counts = df["is_active"].value_counts()

    fig, ax = plt.subplots()

    ax.pie(
        counts.values,
        labels=["Active" if x else "Inactive" for x in counts.index],
        autopct="%1.1f%%"
    )

    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.write("Cancellation Reasons")

    reasons = df["cancel_reason"].value_counts()

    fig, ax = plt.subplots()

    reasons.plot(kind="bar", ax=ax)
    ax.set_xlabel("Reason")
    ax.set_ylabel("Members")
    plt.xticks(rotation=45, ha="right")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Plan Distribution
st.subheader("Membership Plan Distribution")

fig, ax = plt.subplots()

df["plan_name"].value_counts().plot(kind="bar", ax=ax)

ax.set_xlabel("Membership Plan")
ax.set_ylabel("Number of Members")

plt.xticks(rotation=30, ha="right")
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

# Member Analysis
st.subheader("Member Analysis")

safe_columns = [
    "plan_name", "price", "location_name",
    "is_active", "total_visits", "cancel_reason"
]

st.dataframe(df[safe_columns], use_container_width=True)
# Business Insights

st.divider()
st.subheader("Business Insights")

# Most common cancellation reason
top_reason = df["cancel_reason"].value_counts().idxmax()

# Most popular membership plan
popular_plan = df["plan_name"].value_counts().idxmax()

# Inactive member percentage
inactive_percentage = (
    (~df["is_active"]).sum() / len(df)
) * 100

col1, col2 = st.columns(2)

with col1:
    st.write("### Key Findings")
    st.write(f"Most common cancellation reason: {top_reason}")
    st.write(f"Most popular membership plan: {popular_plan}")
    st.write(f"Inactive members: {inactive_percentage:.1f}%")

with col2:
    st.write("### Business Recommendations")
    st.write("1. Improve services based on cancellation feedback.")
    st.write("2. Introduce engagement programs for inactive members.")
    st.write("3. Review membership plans to improve retention.")
# Prediction
st.divider()
st.subheader("Member Status Prediction")

total_visits = st.number_input("Total Visits", min_value=0, value=10)
price = st.number_input("Membership Price", min_value=0.0, value=500.0)
share = st.number_input("Plan Share", min_value=0.0, value=1.0)
capacity = st.number_input("Location Capacity", min_value=0, value=100)

has_pool = st.selectbox("Has Pool?", [True, False])

plan_name = st.selectbox(
    "Plan",
    sorted(df["plan_name"].dropna().unique())
)

location_name = st.selectbox(
    "Location",
    sorted(df["location_name"].dropna().unique())
)

if st.button("Predict Member Status"):

    input_data = pd.DataFrame([{
        "total_visits": total_visits,
        "price": price,
        "share": share,
        "capacity": capacity,
        "has_pool": has_pool,
        "plan_name": plan_name,
        "location_name": location_name
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("Predicted Status: Churned")
    else:
        st.success("Predicted Status: Active")

st.caption(
    "Note: This model classifies existing membership status. "
    "It does not predict future cancellations."
)
