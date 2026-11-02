"""Streamlit app for the Algerian Forest Fires FWI prediction project.

Put this file in the project root (next to the `models/` folder) and run:
    streamlit run streamlit_app.py
"""
import pickle
from pathlib import Path

import streamlit as st

MODEL_DIR = Path(__file__).parent / "models"

st.set_page_config(page_title="FWI Prediction", page_icon="🔥", layout="centered")


@st.cache_resource
def load_models():
    """Load the ridge model and the standard scaler once and reuse them."""
    with open(MODEL_DIR / "ridge.pkl", "rb") as f:
        ridge = pickle.load(f)
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return ridge, scaler


try:
    ridge_model, standard_scaler = load_models()
except FileNotFoundError:
    st.error(
        "Could not find `models/ridge.pkl` and `models/scaler.pkl`. "
        "Place this file in the project root, next to the `models` folder."
    )
    st.stop()

# Same encoding as used when the model was trained
REGION_MAP = {"Bejaia": 0, "Sidi Bel-Abbes": 1}
CLASSES_MAP = {"Not fire": 0, "Fire": 1}

st.title("🔥 Forest Fire Weather Index (FWI) Predictor")
st.write(
    "Enter the weather conditions and fire-index values below to predict the "
    "Fire Weather Index. The model is a Ridge regression trained on the "
    "Algerian Forest Fires dataset."
)

with st.form("fwi_form"):
    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input(
            "Temperature (°C)", min_value=0.0, max_value=50.0, value=30.0, step=0.5
        )
        rh = st.number_input(
            "Relative humidity, RH (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0
        )
        ws = st.number_input(
            "Wind speed, Ws (km/h)", min_value=0.0, max_value=60.0, value=15.0, step=1.0
        )
        rain = st.number_input(
            "Rain (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.1
        )
        region_label = st.selectbox("Region", list(REGION_MAP.keys()))

    with col2:
        ffmc = st.number_input(
            "FFMC (Fine Fuel Moisture Code)", min_value=0.0, max_value=100.0, value=80.0, step=0.5
        )
        dmc = st.number_input(
            "DMC (Duff Moisture Code)", min_value=0.0, max_value=100.0, value=10.0, step=0.5
        )
        isi = st.number_input(
            "ISI (Initial Spread Index)", min_value=0.0, max_value=30.0, value=3.0, step=0.1
        )
        classes_label = st.selectbox("Classes (fire occurred?)", list(CLASSES_MAP.keys()))

    submitted = st.form_submit_button("Predict FWI", type="primary", use_container_width=True)

if submitted:
    classes = CLASSES_MAP[classes_label]
    region = REGION_MAP[region_label]

    # Feature order must match the order used during training
    features = [[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]]

    try:
        scaled = standard_scaler.transform(features)
        prediction = float(ridge_model.predict(scaled)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    else:
        st.metric("Predicted FWI", f"{prediction:.2f}")

with st.expander("About the inputs"):
    st.markdown(
        """
- **FFMC**: Fine Fuel Moisture Code, the moisture of litter and fine fuels.
- **DMC**: Duff Moisture Code, the moisture of loosely compacted organic layers.
- **ISI**: Initial Spread Index, the expected rate of fire spread.
- **Classes**: whether a fire occurred (Not fire = 0, Fire = 1).
- **Region**: Bejaia = 0, Sidi Bel-Abbes = 1.
        """
    )
    st.caption(
        "The model learned from data in a limited range, so values far outside "
        "typical Algerian summer conditions may give unreliable results."
    )