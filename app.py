# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Optical Systems ML UI", layout="centered")

st.title("🔬 Optical Systems — ML Predictor")
st.write("Predict BER (regression) or System Health (classification) from optical parameters.")

# -------------------------
# Sidebar / options
# -------------------------
task = st.sidebar.selectbox("Task", ["Regression — BER_Downstream", "Classification — SystemHealth"])
show_shap = st.sidebar.checkbox("Show SHAP explanation", value=False)

# Pick model file based on task
model_file = "best_regression_model.pkl" if task.startswith("Regression") else "best_classification_model.pkl"

if not os.path.exists(model_file):
    st.error(f"Model file not found: `{model_file}`. Save the pipeline file next to app.py.")
    st.stop()

# Load pipeline (includes preprocessor + model)
pipeline = joblib.load(model_file)

st.success(f"Loaded model: `{model_file}`")

# -------------------------
# Input widgets (physical params)
# -------------------------
st.subheader("Input optical parameters")

col1, col2 = st.columns(2)
with col1:
    input_power = st.number_input("Input Power (units)", min_value=0.01, max_value=100.0, value=1.0, step=0.1, format="%.3f")
    dispersion = st.number_input("Dispersion (ps/nm) / domain units", min_value=0.0, max_value=500.0, value=10.0, step=0.1)
    refr_index = st.number_input("Refractive Index", min_value=1.0, max_value=3.0, value=1.45, step=0.001, format="%.3f")
with col2:
    curvature = st.number_input("Curvature (radius units)", min_value=0.1, max_value=500.0, value=25.0, step=0.1)
    wavelength = st.number_input("Wavelength (µm)", min_value=0.1, max_value=5.0, value=1.45, step=0.001, format="%.3f")

# -------------------------
# Derived features (same formulas as training)
# -------------------------
def compute_derived(input_power, dispersion, refr_index, curvature):
    attenuation_dB = 10 * np.log10(input_power / (input_power + np.random.uniform(0.01, 1.0)))
    focal_length = curvature / (2 * (refr_index - 1))
    # note: we don't compute BER here because it's the target in regression
    return float(attenuation_dB), float(focal_length)

# Provide a button so derived randomness is only applied on click
if st.button("Prepare inputs & predict"):
    att_dB, focal_len = compute_derived(input_power, dispersion, refr_index, curvature)

    # Build input dataframe with the same columns names your pipeline expects
    user_df = pd.DataFrame([{
        "InputPower": input_power,
        "Dispersion": dispersion,
        "RefractiveIndex": refr_index,
        "Curvature": curvature,
        "Wavelength": wavelength,
        "Attenuation_dB": att_dB,
        "FocalLength": focal_len
    }])

    st.markdown("**Input summary (used for prediction):**")
    st.dataframe(user_df.T.rename(columns={0: "value"}))

    # Predict with pipeline (pipeline contains preprocessor)
    try:
        pred = pipeline.predict(user_df)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    if task.startswith("Regression"):
        # regression -> single numeric value (BER)
        ber_pred = float(pred[0])
        st.success(f"🔮 Predicted BER_Downstream: **{ber_pred:.6e}**")
    else:
        # classification -> pipeline's model likely returns numeric labels (0/1/2).
        # Map to strings. Default mapping based on training label order: ['Good','Moderate','Poor']
        mapping = {0: "Good", 1: "Moderate", 2: "Poor"}
        try:
            numeric = int(pred[0])
            label = mapping.get(numeric, str(numeric))
            st.success(f"🔮 Predicted SystemHealth: **{label}** (raw: {numeric})")
        except Exception:
            # If model returns strings directly
            label = str(pred[0])
            st.success(f"🔮 Predicted SystemHealth: **{label}**")

    # Optional SHAP explanation
    if show_shap:
        st.subheader("SHAP explanation (summary)")
        try:
            # pipeline structure assumed: ('preprocessing', ...), ('model', estimator)
            preproc = pipeline.named_steps.get("preprocessing") or pipeline.named_steps.get("preproc") or pipeline.named_steps.get("preprocessor")
            model_est = pipeline.named_steps.get("model") or pipeline.named_steps.get("regressor") or pipeline.named_steps.get("clf")

            if preproc is None or model_est is None:
                st.warning("Could not find preprocessing/model steps in pipeline for SHAP. Skipping SHAP.")
            else:
                X_trans = preproc.transform(user_df)
                explainer = shap.Explainer(model_est)
                shap_values = explainer(X_trans)

                # summary plot (matplotlib)
                plt.figure(figsize=(8, 4))
                shap.summary_plot(shap_values, features=X_trans, feature_names=preproc.get_feature_names_out(), show=False)
                st.pyplot(plt.gcf())
                plt.clf()
        except Exception as e:
            st.error(f"SHAP failed: {e}")

st.markdown("---")

