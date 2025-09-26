import streamlit as st
import numpy as np

st.set_page_config(
    page_title="XO - Exoplanet Habitability Classifier",
    page_icon="🌍",
    layout="centered",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    .stApp { background-color: #0a0a0a; color: #ffffff; }
    .main .block-container { padding-top: 3rem; padding-bottom: 3rem; max-width: 1000px; background-color: #0a0a0a; }

    .main-title { font-family: 'Playfair Display', serif; font-size: 5rem; font-weight: 600; text-align: center; color: #ffffff; letter-spacing: -2px; line-height: 1.1; margin: 2rem 0; text-shadow: 0 2px 4px rgba(255,255,255,0.1); }
    .main-subtitle { font-family: 'Inter', sans-serif; font-size: 1.1rem; font-weight: 400; text-align: center; color: #cccccc; margin-bottom: 2.5rem; letter-spacing: 2px; text-transform: uppercase; }

    .section-title { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 500; color: #ffffff; margin: 1.5rem 0; text-align: center; letter-spacing: -0.5px; }

    .result-container { background: #1a1a1a; border: 2px solid #444444; border-radius: 16px; padding: 2.2rem; margin: 2rem 0; text-align: center; box-shadow: 0 12px 40px rgba(255,255,255,0.05); }
    .result-title { font-family: 'Playfair Display', serif; font-size: 2.6rem; font-weight: 600; color: #ffffff; margin-bottom: 0.5rem; letter-spacing: -1px; text-shadow: 0 2px 4px rgba(255,255,255,0.1); }
    .result-score { font-family: 'Inter', sans-serif; font-size: 1.1rem; font-weight: 400; color: #cccccc; margin-bottom: 1rem; }

    .prediction-source { background: #2a2a2a; border: 1px solid #555555; border-left: 4px solid #667eea; padding: 1rem 1.4rem; margin: 1.2rem 0; border-radius: 8px; }
    .source-text { font-family: 'Inter', sans-serif; font-size: 1rem; color: #ffffff; margin: 0; font-weight: 500; }

    .factor-item { background: #2a2a2a; border-radius: 12px; padding: 1.1rem 1.4rem; margin: 0.6rem 0; border-left: 4px solid #666666; }
    .factor-positive { border-left-color: #28a745; background: #1a2f1a; color: #90ee90; }
    .factor-warning { border-left-color: #ffc107; background: #2f2a1a; color: #ffd700; }
    .factor-negative { border-left-color: #dc3545; background: #2f1a1a; color: #ff6b6b; }
    .factor-item strong { color: #ffffff; font-size: 1.05rem; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        import joblib
        model = joblib.load('models/champion_random_forest.joblib')
        return model, "loaded"
    except Exception:
        return None, "demo"

def calculate_habitable_zone(stellar_mass: float, stellar_temp: float):
    luminosity = (stellar_mass ** 3.5) * ((stellar_temp / 5778) ** 4)
    hz_inner = 0.95 * np.sqrt(luminosity)
    hz_outer = 1.37 * np.sqrt(luminosity)
    return float(hz_inner), float(hz_outer), float(luminosity)

def prepare_features_for_model(pl_rade, pl_orbsmax, st_teff, st_mass, pl_eqt=None):
    if pl_eqt is None:
        luminosity = (st_mass ** 3.5) * ((st_teff / 5778) ** 4)
        pl_eqt = 278 * np.sqrt(luminosity) / np.sqrt(pl_orbsmax)

    pl_bmasse = pl_rade ** 2.06
    stellar_luminosity = (st_mass ** 3.5) * ((st_teff / 5778) ** 4)
    hz_inner, hz_outer, _ = calculate_habitable_zone(st_mass, st_teff)
    hz_position = pl_orbsmax / np.sqrt(stellar_luminosity)
    in_habitable_zone = 1 if hz_inner <= pl_orbsmax <= hz_outer else 0

    esi_radius = 1 - abs(pl_rade - 1) / (pl_rade + 1)
    esi_mass = 1 - abs(pl_bmasse - 1) / (pl_bmasse + 1)
    esi_temperature = 1 - abs(pl_eqt - 288) / (pl_eqt + 288)
    esi_surface = (esi_radius + esi_temperature) / 2

    escape_velocity_ratio = float(np.sqrt(pl_bmasse) / pl_rade)
    stellar_flux = float(stellar_luminosity / (pl_orbsmax ** 2))

    habitability_score = 0
    if in_habitable_zone:
        habitability_score += 3
    if 0.5 <= pl_rade <= 2.0:
        habitability_score += 2
    if 250 <= pl_eqt <= 350:
        habitability_score += 2
    if esi_surface > 0.7:
        habitability_score += 1
    if 0.3 <= st_mass <= 1.5:
        habitability_score += 1
    if escape_velocity_ratio >= 1.0:
        habitability_score += 1

    features = np.array([
        pl_rade, pl_bmasse, pl_orbsmax, st_teff, st_mass, pl_eqt,
        stellar_luminosity, hz_position, in_habitable_zone,
        esi_radius, esi_mass, esi_temperature, esi_surface,
        escape_velocity_ratio, stellar_flux, habitability_score,
    ]).reshape(1, -1)

    return features

def assess_habitability_physics(pl_rade, pl_orbsmax, st_teff, st_mass, pl_eqt=None):
    if pl_eqt is None:
        luminosity = (st_mass ** 3.5) * ((st_teff / 5778) ** 4)
        pl_eqt = float(278 * np.sqrt(luminosity) / np.sqrt(pl_orbsmax))

    hz_inner, hz_outer, luminosity = calculate_habitable_zone(st_mass, st_teff)

    score = 0
    factors = []

    if hz_inner <= pl_orbsmax <= hz_outer:
        score += 40
        factors.append(("positive", "Located in Habitable Zone", f"{hz_inner:.3f}–{hz_outer:.3f} AU envelope"))
    else:
        if pl_orbsmax < hz_inner:
            factors.append(("negative", "Too Close to Star", f"Inside HZ (min {hz_inner:.3f} AU)"))
        else:
            factors.append(("negative", "Too Far from Star", f"Outside HZ (max {hz_outer:.3f} AU)"))

    if 0.8 <= pl_rade <= 1.2:
        size_score = 25
        factors.append(("positive", "Earth-like Size", f"{pl_rade:.2f} R⊕"))
    elif 0.5 <= pl_rade <= 2.0:
        size_score = 20
        factors.append(("positive", "Good Size Range", f"{pl_rade:.2f} R⊕"))
    elif pl_rade < 0.5:
        size_score = 5
        factors.append(("warning", "Very Small Planet", f"{pl_rade:.2f} R⊕"))
    else:
        size_score = 8
        factors.append(("warning", "Large Planet", f"{pl_rade:.2f} R⊕"))
    score += size_score

    if 273 <= pl_eqt <= 313:
        temp_score = 25
        factors.append(("positive", "Near Earth-like Temperature", f"{pl_eqt:.0f} K"))
    elif 250 <= pl_eqt <= 350:
        temp_score = 20
        factors.append(("positive", "Broad Temperate Range", f"{pl_eqt:.0f} K"))
    else:
        temp_score = 0
        if pl_eqt < 250:
            factors.append(("negative", "Too Cold", f"{pl_eqt:.0f} K"))
        else:
            factors.append(("negative", "Too Hot", f"{pl_eqt:.0f} K"))
    score += temp_score

    stellar_score = 0
    if 3500 <= st_teff <= 6500:
        stellar_score += 5
        factors.append(("positive", "Stable Star Temperature", f"{st_teff:.0f} K"))

    if 0.5 <= st_mass <= 1.2:
        stellar_score += 5
        factors.append(("positive", "Favorable Star Mass", f"{st_mass:.2f} M☉"))

    score += stellar_score

    if score >= 80:
        category = "HIGHLY PROMISING"
    elif score >= 65:
        category = "POTENTIALLY HABITABLE"
    elif score >= 45:
        category = "MARGINAL HABITABILITY"
    else:
        category = "NOT HABITABLE"

    return {
        "score": int(score),
        "category": category,
        "factors": factors,
        "hz_inner": hz_inner,
        "hz_outer": hz_outer,
        "pl_eqt": pl_eqt,
        "luminosity": luminosity,
    }

def main():
    st.markdown(
        """
        <div class="main-title">EXOPLANET<br/>HABITABILITY<br/>CLASSIFIER</div>
        <div class="main-subtitle">AI‑Powered Astronomical Discovery</div>
        """,
        unsafe_allow_html=True,
    )

    model, model_status = load_model()
    # Model status banner removed to avoid empty/extra boxes

    with st.form("habitability_analysis"):
        st.markdown('<div class="section-title">Planetary Parameters</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            pl_rade = st.slider("Planet Radius (Earth radii)", 0.1, 10.0, 1.0, 0.1)
            use_custom_temp = st.checkbox("Specify temperature manually")
        with col2:
            pl_orbsmax = st.slider("Orbital Distance (AU)", 0.01, 5.0, 1.0, 0.01)
            if use_custom_temp:
                pl_eqt = st.slider("Equilibrium Temperature (K)", 100, 1000, 288, 5)
            else:
                pl_eqt = None

        st.markdown('<div class="section-title">Stellar Parameters</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st_teff = st.slider("Stellar Temperature (K)", 2000, 8000, 5778, 25)
        with col2:
            st_mass = st.slider("Stellar Mass (Solar masses)", 0.1, 3.0, 1.0, 0.05)

        submitted = st.form_submit_button("ANALYZE HABITABILITY")

    if not submitted:
        return

    physics_result = assess_habitability_physics(pl_rade, pl_orbsmax, st_teff, st_mass, pl_eqt)

    ai_result = None
    ai_confidence = None
    if model is not None and model_status == "loaded":
        try:
            features = prepare_features_for_model(pl_rade, pl_orbsmax, st_teff, st_mass, physics_result["pl_eqt"])
            ai_prediction = model.predict(features)[0]
            ai_probabilities = model.predict_proba(features)[0]
            ai_confidence = float(np.max(ai_probabilities) * 100)
            ai_result = {"category": "POTENTIALLY HABITABLE" if ai_prediction == 1 else "NOT HABITABLE", "confidence": ai_confidence}
        except Exception as e:
            st.error(f"AI Model Error: {str(e)}")

    # Result container only when results exist
    if ai_result:
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="result-title">{ai_result["category"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-score">Model confidence: {ai_confidence:.1f}% · Physics score: {physics_result["score"]}/100</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="result-title">{physics_result["category"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-score">Habitability score: {physics_result["score"]}/100 (physics-based)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="prediction-source">', unsafe_allow_html=True)
    if ai_result:
        st.markdown('<p class="source-text"><strong>PREDICTION SOURCE:</strong> Model + physics analysis</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="source-text"><strong>PREDICTION SOURCE:</strong> Physics-based analysis only</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Analysis Factors</div>', unsafe_allow_html=True)
    for factor_type, factor_name, factor_detail in physics_result["factors"]:
        factor_class = f"factor-item factor-{factor_type}"
        st.markdown(f'<div class="{factor_class}"><strong>{factor_name}</strong><br/>{factor_detail}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
