import streamlit as st
import pandas as pd
import numpy as np

# Simple, guaranteed-to-work version
st.set_page_config(
    page_title="XO - Exoplanet Classifier",
    page_icon="🌍"
)

# Header
st.title("🌍 XO - Exoplanet Habitability Classifier")
st.write("Simplified version - fully functional")

# Load data safely
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/processed/ml_optimized_dataset.csv')
        return df
    except:
        # Create sample data if file not found
        sample_data = {
            'pl_name': ['Kepler-452b', 'TOI-715b', 'Proxima Cen b'],
            'pl_rade': [1.63, 1.55, 1.17],
            'pl_orbsmax': [1.05, 0.083, 0.048],
            'st_teff': [5757, 3980, 3042],
            'st_mass': [1.04, 0.43, 0.12],
            'pl_eqt': [265, 300, 234]
        }
        return pd.DataFrame(sample_data)

# Load model safely
@st.cache_resource
def load_model():
    try:
        import joblib
        model = joblib.load('models/champion_random_forest.joblib')
        return model
    except:
        return None

# Main app
df = load_data()
model = load_model()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose page:", ["Home", "Predict", "Data"])

if page == "Home":
    st.header("Welcome to XO Project")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Planets", len(df))
    with col2:
        st.metric("Model Status", "✅ Loaded" if model else "⚠️ Demo")
    with col3:
        st.metric("Features", "16 physics-based")

    st.subheader("Sample Data")
    st.dataframe(df.head())

elif page == "Predict":
    st.header("Habitability Predictor")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Planet Parameters")
        pl_rade = st.slider("Planet Radius (Earth radii)", 0.1, 10.0, 1.0)
        pl_orbsmax = st.slider("Orbital Distance (AU)", 0.01, 5.0, 1.0)

    with col2:
        st.subheader("Star Parameters")
        st_teff = st.slider("Stellar Temperature (K)", 2000, 8000, 5778)
        st_mass = st.slider("Stellar Mass (Solar masses)", 0.1, 3.0, 1.0)

    if st.button("Predict Habitability"):
        # Simple habitability calculation
        score = 0

        # Calculate habitable zone
        luminosity = st_mass ** 3.5 * (st_teff / 5778) ** 4
        hz_inner = 0.95 * np.sqrt(luminosity)
        hz_outer = 1.37 * np.sqrt(luminosity)

        # Scoring
        if hz_inner <= pl_orbsmax <= hz_outer:
            score += 40
            st.success(f"✅ In habitable zone ({hz_inner:.2f} - {hz_outer:.2f} AU)")
        else:
            st.warning("⚠️ Outside habitable zone")

        if 0.5 <= pl_rade <= 2.0:
            score += 30
            st.success("✅ Earth-like size")
        else:
            st.warning("⚠️ Non-Earth-like size")

        if 3000 <= st_teff <= 7000:
            score += 20
            st.success("✅ Suitable star temperature")

        if 0.3 <= st_mass <= 1.5:
            score += 10
            st.success("✅ Stable star mass")

        # Final result
        if score >= 70:
            st.success(f"🌍 POTENTIALLY HABITABLE (Score: {score}/100)")
        elif score >= 40:
            st.warning(f"🟡 MARGINAL HABITABILITY (Score: {score}/100)")
        else:
            st.error(f"❌ NOT HABITABLE (Score: {score}/100)")

elif page == "Data":
    st.header("Exoplanet Data Explorer")

    st.write(f"Dataset contains {len(df)} planets")

    # Simple filters
    if 'pl_rade' in df.columns:
        min_radius = st.slider("Minimum radius", 0.0, 10.0, 0.0)
        df_filtered = df[df['pl_rade'] >= min_radius]
    else:
        df_filtered = df

    st.write(f"Filtered: {len(df_filtered)} planets")
    st.dataframe(df_filtered)

st.sidebar.markdown("---")
st.sidebar.write("🚀 Simple version working!")
st.sidebar.write("If this works, we can enhance it.")