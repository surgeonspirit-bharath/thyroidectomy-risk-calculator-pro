import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Thyroidectomy Risk Pro", layout="wide")

# --- STYLE ---
st.markdown("""
<style>
.card {
    padding:20px;
    border-radius:12px;
    background-color:#f7f9fc;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.title("🩺 Thyroidectomy Risk Pro (Advanced)")

# --- INPUTS ---
st.subheader("Patient Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    strain = st.slider("Strain Elastography", 1, 4, 2)
    fnac = st.selectbox("FNAC", ["Benign", "Malignant"])

with col2:
    retro = st.selectbox("Retrosternal Extension", ["No", "Yes"])
    reop = st.selectbox("Reoperative Surgery", ["No", "Yes"])

with col3:
    tirads = st.selectbox("ACR TIRADS", [2,3,4,5])
    thyroiditis = st.selectbox("Thyroiditis", ["No", "Yes"])
    gland = st.slider("Gland Size (ml)", 10, 150, 50)

# --- ENCODING ---
fnac_val = 1 if fnac == "Malignant" else 0
retro_val = 1 if retro == "Yes" else 0
reop_val = 1 if reop == "Yes" else 0
thyroiditis_val = 1 if thyroiditis == "Yes" else 0

# --- MODEL COEFFICIENTS ---
beta_0 = -3.0
beta_strain = 0.38
beta_fnac = 0.99
beta_retro = 1.42
beta_reop = 0.36

# NEW moderate predictors
beta_tirads = 0.25
beta_thyroiditis = 0.15
beta_gland = 0.005  # per ml

# --- LOGISTIC MODEL ---
logit = (
    beta_0 +
    beta_strain * strain +
    beta_fnac * fnac_val +
    beta_retro * retro_val +
    beta_reop * reop_val +
    beta_tirads * tirads +
    beta_thyroiditis * thyroiditis_val +
    beta_gland * gland
)

prob = 1 / (1 + np.exp(-logit))

# --- DISPLAY ---
st.markdown("---")

colA, colB = st.columns([2,1])

with colA:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.subheader("📊 Risk Output")
    st.progress(float(prob))
    st.markdown(f"### Risk: **{prob*100:.1f}%**")

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
    - Strain: **{strain}**
    - FNAC: **{fnac}**
    - Retrosternal: **{retro}**
    - Reoperative: **{reop}**
    - TIRADS: **{tirads}**
    - Thyroiditis: **{thyroiditis}**
    - Gland Size: **{gland} ml**
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- REPORT ---
st.markdown("---")

if st.button("📄 Download Report"):
    report = pd.DataFrame({
        "Parameter": ["Strain","FNAC","Retrosternal","Reoperative","TIRADS","Thyroiditis","Gland Size","Risk (%)"],
        "Value": [strain, fnac, retro, reop, tirads, thyroiditis, gland, round(prob*100,1)]
    })

    csv = report.to_csv(index=False)
    st.download_button("Download CSV", csv, "thyroid_risk_report.csv")

# --- INFO ---
with st.expander("ℹ️ Model Details"):
    st.write("""
    Strong predictors:
    - FNAC malignancy
    - Retrosternal extension
    - Elastography
    
    Moderate:
    - TIRADS
    - Reoperative surgery
    
    Optional:
    - Thyroiditis
    - Gland size
    """)

st.caption("For research use only")
