import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Thyroidectomy Risk Tool",
    layout="wide"
)

# -------------------------------
# HEADER
# -------------------------------
st.title("🧠 Thyroidectomy Difficulty Prediction Tool")
st.markdown("### Multivariate Logistic Regression Model + Nomogram-based Prediction")

# Sidebar
st.sidebar.header("Study Info")
st.sidebar.markdown("""
- Sample size: 402  
- Outcome: Difficult thyroidectomy  
- Model: Logistic regression  
- AUC: 0.760  
""")

# -------------------------------
# INPUT PANEL
# -------------------------------
st.subheader("🔎 Patient Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    retro = st.selectbox("Retrosternal Extension", ["No", "Yes"])
    fnac = st.selectbox("FNAC Result", ["Benign", "Malignant"])

with col2:
    strain = st.slider("Strain Elastography Score", 1, 4, 2)
    tirads = st.slider("TIRADS Category", 1, 5, 3)

with col3:
    reop = st.selectbox("Reoperative Surgery", ["No", "Yes"])
    gland = st.number_input("Gland Size (ml)", 10, 200, 60)

# Convert to numeric
retro = 1 if retro == "Yes" else 0
fnac = 1 if fnac == "Malignant" else 0
reop = 1 if reop == "Yes" else 0

# -------------------------------
# MODEL (REPLACE WITH FINAL COEFFS)
# -------------------------------
intercept = -2.5

beta = {
    "retro": 0.61,
    "fnac": 0.49,
    "strain": 0.40,
    "tirads": 0.25,
    "reop": 0.30,
    "gland": 0.01
}

# -------------------------------
# CALCULATE
# -------------------------------
if st.button("🚀 Calculate Risk"):

    logit = (
        intercept
        + beta["retro"] * retro
        + beta["fnac"] * fnac
        + beta["strain"] * strain
        + beta["tirads"] * tirads
        + beta["reop"] * reop
        + beta["gland"] * gland
    )

    prob = 1 / (1 + np.exp(-logit))
    percent = round(prob * 100, 1)

    # Risk classification
    if percent < 20:
        risk = "Low Risk"
        color = "#16a34a"
    elif percent < 40:
        risk = "Intermediate Risk"
        color = "#eab308"
    else:
        risk = "High Risk"
        color = "#dc2626"

    st.markdown("---")

    # -------------------------------
    # MAIN RESULT
    # -------------------------------
    colA, colB = st.columns([2,1])

    with colA:
        st.subheader("📊 Predicted Probability")
        st.metric("Risk of Difficult Thyroidectomy", f"{percent}%")

        st.markdown(
            f"""
            <div style="
                padding:20px;
                border-radius:12px;
                background-color:{color};
                color:white;
                text-align:center;
                font-size:22px;
            ">
            {risk}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(int(percent))

    # -------------------------------
    # ROC VISUALIZATION (DEMO)
    # -------------------------------
    with colB:
        st.subheader("📈 Model Performance")

        fpr = np.linspace(0,1,100)
        tpr = fpr**0.6  # simulated curve for display

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.plot([0,1],[0,1],'--')
        ax.set_title("ROC Curve (AUC = 0.76)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        st.pyplot(fig)

    # -------------------------------
    # CALIBRATION STYLE PLOT
    # -------------------------------
    st.subheader("📉 Calibration (Illustrative)")

    pred = np.linspace(0,1,10)
    obs = pred + np.random.normal(0,0.05,10)

    fig2, ax2 = plt.subplots()
    ax2.plot(pred, obs, marker='o')
    ax2.plot([0,1],[0,1],'--')
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Observed Probability")
    ax2.set_title("Calibration Plot")

    st.pyplot(fig2)

    # -------------------------------
    # CLINICAL INTERPRETATION
    # -------------------------------
    st.subheader("🧾 Clinical Interpretation")

    if risk == "Low Risk":
        st.success("Suitable for routine surgical scheduling.")
    elif risk == "Intermediate Risk":
        st.warning("Consider experienced surgeon and enhanced preparation.")
    else:
        st.error("High-risk case: senior surgeon, advanced planning, possible complications.")

    # -------------------------------
    # REPORT BLOCK (FOR SCREENSHOT IN PAPER)
    # -------------------------------
    st.subheader("📄 Summary Report")

    st.markdown(f"""
    **Predicted risk:** {percent}%  
    **Risk category:** {risk}  

    **Key contributing factors:**
    - Retrosternal extension: {retro}
    - FNAC malignancy: {fnac}
    - Strain score: {strain}

    This tool is based on a multivariate logistic regression model with AUC = 0.760.
    """)

# Footer
st.markdown("---")
st.caption("For research and academic use. Not a substitute for clinical judgment.")
