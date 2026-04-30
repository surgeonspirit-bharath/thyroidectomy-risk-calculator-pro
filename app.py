import streamlit as st
import numpy as np
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Thyroidectomy Risk Pro",
    layout="wide",
    page_icon="🩺"
)

# --- CUSTOM CSS (makes it look premium) ---
st.markdown("""
<style>
.big-font {font-size:22px !important;}
.card {
    padding:20px;
    border-radius:12px;
    background-color:#f7f9fc;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("🩺 Thyroidectomy Risk Pro")
st.markdown("Advanced clinical decision support tool")

# --- INPUT SECTION ---
st.subheader("Patient Parameters")

col1, col2 = st.columns(2)

with col1:
    strain = st.slider("Strain Elastography Score", 1, 4, 2)
    fnac = st.selectbox("FNAC Result", ["Benign", "Malignant"])

with col2:
    retro = st.selectbox("Retrosternal Extension", ["No", "Yes"])
    reop = st.selectbox("Reoperative Surgery", ["No", "Yes"])

# --- ENCODING ---
fnac_val = 1 if fnac == "Malignant" else 0
retro_val = 1 if retro == "Yes" else 0
reop_val = 1 if reop == "Yes" else 0

# --- MODEL ---
beta_0 = -3.0
beta_strain = 0.38
beta_fnac = 0.99
beta_retro = 1.42
beta_reop = 0.36

logit = (
    beta_0 +
    beta_strain * strain +
    beta_fnac * fnac_val +
    beta_retro * retro_val +
    beta_reop * reop_val
)

prob = 1 / (1 + np.exp(-logit))

# --- DISPLAY ---
st.markdown("---")

colA, colB = st.columns([2,1])

with colA:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.subheader("📊 Risk Output")
    st.progress(float(prob))
    st.markdown(f"<p class='big-font'>Risk: <b>{prob*100:.1f}%</b></p>", unsafe_allow_html=True)

    if prob < 0.2:
        st.success("🟢 Low Risk")
    elif prob < 0.5:
        st.warning("🟡 Moderate Risk")
    else:
        st.error("🔴 High Risk")

    st.markdown('</div>', unsafe_allow_html=True)

with colB:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.subheader("🧠 Summary")
    st.write(f"""
    - Strain Score: **{strain}**
    - FNAC: **{fnac}**
    - Retrosternal: **{retro}**
    - Reoperative: **{reop}**
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- REPORT GENERATION ---
st.markdown("---")
st.subheader("📄 Generate Clinical Report")

if st.button("Download Report"):
    report = pd.DataFrame({
        "Parameter": ["Strain", "FNAC", "Retrosternal", "Reoperative", "Risk (%)"],
        "Value": [strain, fnac, retro, reop, round(prob*100,1)]
    })

    csv = report.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="thyroidectomy_risk_report.csv",
        mime="text/csv"
    )

# --- INFO SECTION ---
with st.expander("ℹ️ About Model"):
    st.write("""
    This model predicts difficult thyroidectomy using:
    - Elastography
    - FNAC cytology
    - Anatomical factors

    Based on multivariate logistic regression.
    """)

# --- FOOTER ---
st.markdown("---")
st.caption("Clinical decision support tool | For research use")
