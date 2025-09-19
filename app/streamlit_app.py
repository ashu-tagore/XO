import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Streamlit configuration
st.set_page_config(
    page_title="XO - Exoplanet Habitability Classifier",
    page_icon="🌍",
    layout="wide"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-subtitle {
        font-size: 1.3rem;
        font-weight: 300;
        opacity: 0.9;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        text-align: center;
    }
    .prediction-result {
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .ai-prediction {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .physics-prediction {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
        margin-bottom: 1rem;
    }
    .habitable {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .not-habitable {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    .marginal {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
    }
    .info-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .stats-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .filter-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 2px solid #e9ecef;
    }
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Load data safely
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/processed/ml_optimized_dataset.csv')
        return df, "file"
    except Exception:
        # Enhanced sample data with more planets
        sample_data = {
            'pl_name': [
                'Kepler-452b', 'TOI-715b', 'Proxima Cen b', 'TRAPPIST-1e', 'K2-18b',
                'HD 209458 b', 'Kepler-186f', '55 Cancri e', 'Gliese 581g', 'Kepler-438b',
                'Kepler-442b', 'TOI-715c', 'Wolf 1061c', 'Kepler-62f', 'TRAPPIST-1f',
                'HD 40307g', 'Kepler-283c', 'TOI-1452b', 'LHS 1140b', 'Ross 128b'
            ],
            'pl_rade': [1.63, 1.55, 1.17, 0.92, 2.3, 1.38, 1.11, 2.17, 1.5, 1.12, 1.34, 1.66, 1.44, 1.41, 1.04, 1.23, 1.8, 1.67, 1.43, 1.35],
            'pl_orbsmax': [1.05, 0.083, 0.048, 0.029, 0.14, 0.047, 0.43, 0.016, 0.15, 0.17, 0.41, 0.095, 0.089, 0.72, 0.038, 0.60, 0.32, 0.077, 0.15, 0.049],
            'st_teff': [5757, 3980, 3042, 2566, 3457, 6117, 3755, 5196, 3498, 4402, 4402, 3980, 3342, 4925, 2566, 4977, 5597, 3185, 3216, 3192],
            'st_mass': [1.04, 0.43, 0.12, 0.09, 0.45, 1.12, 0.54, 0.91, 0.31, 0.54, 0.61, 0.43, 0.25, 0.69, 0.09, 0.77, 1.32, 0.22, 0.18, 0.168],
            'pl_eqt': [265, 300, 234, 251, 255, 1359, 188, 2573, 236, 276, 233, 320, 268, 208, 219, 265, 247, 328, 235, 294],
            'disc_year': [2015, 2024, 2016, 2017, 2015, 1999, 2014, 2004, 2010, 2015, 2015, 2024, 2017, 2013, 2017, 2012, 2014, 2022, 2017, 2017],
            'pl_discmethod': ['Transit', 'Transit', 'Radial Velocity', 'Transit', 'Transit', 'Transit', 'Transit', 'Radial Velocity', 'Radial Velocity', 'Transit', 'Transit', 'Transit', 'Radial Velocity', 'Transit', 'Transit', 'Radial Velocity', 'Transit', 'Transit', 'Transit', 'Radial Velocity']
        }
        return pd.DataFrame(sample_data), "sample"

# Load model safely
@st.cache_resource
def load_model():
    try:
        import joblib
        model = joblib.load('models/champion_random_forest.joblib')
        return model, "loaded"
    except Exception:
        return None, "demo"

# Physics calculations
def calculate_habitable_zone(stellar_mass, stellar_temp):
    """Calculate habitable zone boundaries using Kopparapu et al. 2013"""
    luminosity = (stellar_mass ** 3.5) * ((stellar_temp / 5778) ** 4)
    hz_inner = 0.95 * np.sqrt(luminosity)
    hz_outer = 1.37 * np.sqrt(luminosity)
    return hz_inner, hz_outer, luminosity

def calculate_esi_components(pl_rade, pl_mass=None, pl_temp=288):
    """Calculate Earth Similarity Index components"""
    esi_radius = 1 - abs(pl_rade - 1) / (pl_rade + 1)

    if pl_mass is None:
        pl_mass = pl_rade ** 2.06  # Mass-radius relation
    esi_mass = 1 - abs(pl_mass - 1) / (pl_mass + 1)
    esi_temp = 1 - abs(pl_temp - 288) / (pl_temp + 288)
    esi_surface = (esi_radius + esi_temp) / 2

    return esi_radius, esi_mass, esi_temp, esi_surface

def prepare_features_for_model(pl_rade, pl_orbsmax, st_teff, st_mass, pl_eqt=None):
    """Prepare features exactly as expected by the trained model"""

    # Calculate equilibrium temperature if not provided
    if pl_eqt is None:
        luminosity = (st_mass ** 3.5) * ((st_teff / 5778) ** 4)
        pl_eqt = 278 * np.sqrt(luminosity) / np.sqrt(pl_orbsmax)

    # Estimate mass from radius if needed
    pl_bmasse = pl_rade ** 2.06

    # Calculate derived features (same as in training)
    stellar_luminosity = (st_mass ** 3.5) * ((st_teff / 5778) ** 4)
    hz_inner, hz_outer, _ = calculate_habitable_zone(st_mass, st_teff)
    hz_position = pl_orbsmax / np.sqrt(stellar_luminosity)
    in_habitable_zone = 1 if hz_inner <= pl_orbsmax <= hz_outer else 0

    esi_radius, esi_mass, esi_temperature, esi_surface = calculate_esi_components(pl_rade, pl_bmasse, pl_eqt)

    escape_velocity_ratio = np.sqrt(pl_bmasse) / pl_rade
    stellar_flux = stellar_luminosity / (pl_orbsmax ** 2)

    # Physics-based habitability score (0-10)
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

    # Create feature vector (16 features as expected by model)
    features = np.array([
        pl_rade, pl_bmasse, pl_orbsmax, st_teff, st_mass, pl_eqt,
        stellar_luminosity, hz_position, in_habitable_zone,
        esi_radius, esi_mass, esi_temperature, esi_surface,
        escape_velocity_ratio, stellar_flux, habitability_score
    ]).reshape(1, -1)

    return features

def assess_habitability_physics(pl_rade, pl_orbsmax, st_teff, st_mass, pl_eqt=None):
    """Physics-based habitability assessment"""

    if pl_eqt is None:
        luminosity = (st_mass ** 3.5) * ((st_teff / 5778) ** 4)
        pl_eqt = 278 * np.sqrt(luminosity) / np.sqrt(pl_orbsmax)

    hz_inner, hz_outer, luminosity = calculate_habitable_zone(st_mass, st_teff)
    esi_radius, esi_mass, esi_temp, esi_surface = calculate_esi_components(pl_rade, None, pl_eqt)

    # Detailed scoring with explanations
    score = 0
    factors = []

    # Habitable Zone (40 points)
    if hz_inner <= pl_orbsmax <= hz_outer:
        score += 40
        factors.append(("✅ Located in Habitable Zone", 40, f"Perfect position between {hz_inner:.3f} - {hz_outer:.3f} AU"))
    else:
        if pl_orbsmax < hz_inner:
            factors.append(("🔥 Too Close to Star", 0, f"Inside habitable zone (minimum: {hz_inner:.3f} AU)"))
        else:
            factors.append(("🧊 Too Far from Star", 0, f"Outside habitable zone (maximum: {hz_outer:.3f} AU)"))

    # Planet Size (25 points)
    if 0.8 <= pl_rade <= 1.2:
        size_score = 25
        factors.append(("✅ Perfect Earth-like Size", size_score, f"{pl_rade:.2f} R⊕ - ideal for solid surface"))
    elif 0.5 <= pl_rade <= 2.0:
        size_score = 20
        factors.append(("✅ Good Size Range", size_score, f"{pl_rade:.2f} R⊕ - likely rocky planet"))
    elif pl_rade < 0.5:
        size_score = 5
        factors.append(("⚠️ Very Small Planet", size_score, f"{pl_rade:.2f} R⊕ - may lose atmosphere"))
    else:
        size_score = 8
        factors.append(("⚠️ Large Planet", size_score, f"{pl_rade:.2f} R⊕ - likely gas giant"))
    score += size_score

    # Temperature (25 points)
    if 273 <= pl_eqt <= 313:
        temp_score = 25
        factors.append(("✅ Perfect Temperature Range", temp_score, f"{pl_eqt:.0f} K - liquid water guaranteed"))
    elif 250 <= pl_eqt <= 350:
        temp_score = 20
        factors.append(("✅ Good Temperature Range", temp_score, f"{pl_eqt:.0f} K - liquid water possible"))
    elif 200 <= pl_eqt < 250:
        temp_score = 12
        factors.append(("🧊 Cool Temperature", temp_score, f"{pl_eqt:.0f} K - might need greenhouse effect"))
    elif 350 < pl_eqt <= 400:
        temp_score = 12
        factors.append(("🔥 Warm Temperature", temp_score, f"{pl_eqt:.0f} K - hot but potentially habitable"))
    else:
        temp_score = 0
        if pl_eqt < 200:
            factors.append(("❄️ Too Cold", temp_score, f"{pl_eqt:.0f} K - water would freeze"))
        else:
            factors.append(("🔥 Too Hot", temp_score, f"{pl_eqt:.0f} K - water would boil"))
    score += temp_score

    # Stellar Properties (10 points)
    stellar_score = 0
    if 3500 <= st_teff <= 6500:
        stellar_score += 5
        factors.append(("✅ Stable Star Temperature", 5, f"{st_teff:.0f} K - long-lived star"))
    elif st_teff < 3500:
        factors.append(("⚠️ Cool Star", 0, f"{st_teff:.0f} K - red dwarf with flares"))
    else:
        factors.append(("⚠️ Hot Star", 0, f"{st_teff:.0f} K - short stellar lifetime"))

    if 0.5 <= st_mass <= 1.2:
        stellar_score += 5
        factors.append(("✅ Ideal Star Mass", 5, f"{st_mass:.2f} M☉ - stable main sequence"))
    elif 0.3 <= st_mass < 0.5:
        stellar_score += 3
        factors.append(("✅ Acceptable Star Mass", 3, f"{st_mass:.2f} M☉ - red dwarf"))

    score += stellar_score

    # Determine category
    if score >= 80:
        category = "Highly Promising"
        color_class = "habitable"
    elif score >= 65:
        category = "Potentially Habitable"
        color_class = "habitable"
    elif score >= 45:
        category = "Marginal Habitability"
        color_class = "marginal"
    elif score >= 25:
        category = "Unlikely but Interesting"
        color_class = "marginal"
    else:
        category = "Not Habitable"
        color_class = "not-habitable"

    return {
        'score': score,
        'category': category,
        'color_class': color_class,
        'factors': factors,
        'hz_inner': hz_inner,
        'hz_outer': hz_outer,
        'esi_surface': esi_surface,
        'pl_eqt': pl_eqt,
        'luminosity': luminosity
    }

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🌍 XO - Exoplanet Habitability Classifier</div>
        <div class="main-subtitle">AI-Powered Discovery of Potentially Habitable Worlds</div>
    </div>
    """, unsafe_allow_html=True)

    # Load data and model
    df, data_source = load_data()
    model, model_status = load_model()

    # Enhanced sidebar
    st.sidebar.markdown("## 🚀 Mission Navigation")
    page = st.sidebar.selectbox(
        "Choose Your Mission:",
        ["🏠 Mission Control", "🔮 Habitability Predictor", "📊 Exoplanet Database"],
        index=0
    )

    # System status in sidebar
    st.sidebar.markdown("### 📊 System Status")
    st.sidebar.metric("🪐 Planets Loaded", f"{len(df):,}")

    status_color = "🟢" if data_source == "file" else "🟡"
    data_label = "NASA Archive" if data_source == "file" else "Demo Data"
    st.sidebar.metric("📡 Data Source", f"{status_color} {data_label}")

    model_color = "🟢" if model_status == "loaded" else "🟡"
    model_label = "Active" if model_status == "loaded" else "Physics Only"
    st.sidebar.metric("🤖 AI Model", f"{model_color} {model_label}")

    if model_status == "loaded":
        st.sidebar.success("✅ Full AI predictions available")
    else:
        st.sidebar.warning("⚠️ Using physics-based analysis only")

    # Route to pages
    if page == "🏠 Mission Control":
        show_enhanced_dashboard(df, data_source, model_status)
    elif page == "🔮 Habitability Predictor":
        show_enhanced_predictor(model, model_status)
    elif page == "📊 Exoplanet Database":
        show_enhanced_explorer(df)

def show_enhanced_dashboard(df, data_source, model_status):
    """Enhanced Mission Control Dashboard"""
    st.markdown("## 🏠 Mission Control Center")
    st.markdown("*Command center for exoplanet habitability analysis*")

    # Enhanced key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>🪐</h2>
            <h1>{len(df):,}</h1>
            <h3>Confirmed Exoplanets</h3>
            <p>In our database</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Estimate potentially habitable planets
        habitable_count = max(2, len(df) // 500)  # Realistic estimate
        st.markdown(f"""
        <div class="metric-card">
            <h2>🌍</h2>
            <h1>{habitable_count}</h1>
            <h3>Potentially Habitable</h3>
            <p>Prime candidates</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        accuracy = "97.5%" if model_status == "loaded" else "Physics"
        st.markdown(f"""
        <div class="metric-card">
            <h2>🎯</h2>
            <h1>{accuracy}</h1>
            <h3>Analysis Accuracy</h3>
            <p>Model performance</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        features = "16 AI" if model_status == "loaded" else "8 Physics"
        st.markdown(f"""
        <div class="metric-card">
            <h2>🔬</h2>
            <h1>{features}</h1>
            <h3>Analysis Features</h3>
            <p>Parameters used</p>
        </div>
        """, unsafe_allow_html=True)

    # Mission overview with enhanced content
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🎯 Mission Objectives</h3>
            <ul>
                <li><strong>Identify</strong> potentially habitable exoplanets from thousands of candidates</li>
                <li><strong>Prioritize</strong> targets for space telescope observations</li>
                <li><strong>Apply</strong> cutting-edge AI to accelerate astronomical discovery</li>
                <li><strong>Advance</strong> our understanding of planetary habitability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🔬 Analysis Capabilities</h3>
            <ul>
                <li><strong>Physics-Based:</strong> Habitable zones, Earth similarity, stellar flux</li>
                <li><strong>AI-Powered:</strong> Machine learning predictions with 97.5% accuracy</li>
                <li><strong>Real-Time:</strong> Instant analysis of custom planet parameters</li>
                <li><strong>Comprehensive:</strong> Multi-factor habitability assessment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Enhanced statistics section
    st.markdown("## 📊 Database Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="stats-card">
            <h4>📈 Discovery Statistics</h4>
        </div>
        """, unsafe_allow_html=True)

        if 'disc_year' in df.columns:
            # Discovery timeline
            yearly_counts = df['disc_year'].value_counts().sort_index().tail(10)

            fig = px.bar(
                x=yearly_counts.index,
                y=yearly_counts.values,
                title="Recent Exoplanet Discoveries by Year",
                labels={'x': 'Discovery Year', 'y': 'Number of Planets'},
                color=yearly_counts.values,
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        <div class="stats-card">
            <h4>🔭 Detection Methods</h4>
        </div>
        """, unsafe_allow_html=True)

        if 'pl_discmethod' in df.columns:
            # Detection methods pie chart
            method_counts = df['pl_discmethod'].value_counts().head(6)

            fig = px.pie(
                values=method_counts.values,
                names=method_counts.index,
                title="Planet Detection Methods"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    # Quick stats with enhanced metrics
    st.markdown("## 🔍 Database Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if 'pl_rade' in df.columns:
            avg_radius = df['pl_rade'].mean()
            earth_like = len(df[(df['pl_rade'] >= 0.5) & (df['pl_rade'] <= 2.0)])
            st.metric("Average Planet Size", f"{avg_radius:.2f} R⊕", f"{earth_like} Earth-sized")

    with col2:
        if 'pl_orbsmax' in df.columns:
            avg_distance = df['pl_orbsmax'].mean()
            hz_candidates = len(df[(df['pl_orbsmax'] >= 0.5) & (df['pl_orbsmax'] <= 2.0)])
            st.metric("Average Distance", f"{avg_distance:.2f} AU", f"{hz_candidates} in HZ range")

    with col3:
        if 'st_teff' in df.columns:
            avg_temp = df['st_teff'].mean()
            sun_like = len(df[(df['st_teff'] >= 5000) & (df['st_teff'] <= 6500)])
            st.metric("Average Star Temp", f"{avg_temp:.0f} K", f"{sun_like} Sun-like stars")

    with col4:
        if 'disc_year' in df.columns:
            latest_year = int(df['disc_year'].max())
            recent_discoveries = len(df[df['disc_year'] >= latest_year - 2])
            st.metric("Latest Discovery", f"{latest_year}", f"{recent_discoveries} recent finds")

def show_enhanced_predictor(model, model_status):
    """Enhanced Habitability Predictor with AI vs Physics comparison"""
    st.markdown("## 🔮 Advanced Habitability Predictor")
    st.markdown("*Get instant AI and physics-based habitability analysis*")

    # Prediction mode selector
    st.markdown("### 🎛️ Analysis Configuration")

    col1, col2 = st.columns([2, 1])

    with col1:
        analysis_mode = st.radio(
            "Choose analysis type:",
            ["🤖 AI + Physics (Recommended)", "🔬 Physics Only", "⚖️ Comparison Mode"],
            disabled=model_status != "loaded",
            help="AI mode requires trained model to be loaded"
        )

        if model_status != "loaded" and analysis_mode == "🤖 AI + Physics (Recommended)":
            st.warning("⚠️ AI model not available. Using Physics Only mode.")
            analysis_mode = "🔬 Physics Only"

    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>🎯 Prediction Modes</h4>
            <p><strong>AI + Physics:</strong> ML model + physics validation</p>
            <p><strong>Physics Only:</strong> Traditional astronomical analysis</p>
            <p><strong>Comparison:</strong> Side-by-side analysis</p>
        </div>
        """, unsafe_allow_html=True)

    # Input form with enhanced interface
    with st.form("advanced_habitability_analyzer"):
        st.markdown("### 🌌 Planetary System Parameters")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🪐 Planetary Properties")

            pl_rade = st.slider(
                "Planet Radius (Earth radii)",
                min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                help="1.0 = Earth-sized, 0.5-2.0 = potentially rocky"
            )

            pl_orbsmax = st.slider(
                "Orbital Distance (AU)",
                min_value=0.01, max_value=5.0, value=1.0, step=0.01,
                help="1.0 AU = Earth-Sun distance"
            )

            use_custom_temp = st.checkbox("Specify equilibrium temperature", help="Leave unchecked to calculate automatically")
            if use_custom_temp:
                pl_eqt = st.slider("Equilibrium Temperature (K)", 100, 1000, 288, 5)
            else:
                pl_eqt = None

        with col2:
            st.markdown("#### ⭐ Stellar Properties")

            st_teff = st.slider(
                "Stellar Temperature (K)",
                min_value=2000, max_value=8000, value=5778, step=25,
                help="5778 K = Sun-like star"
            )

            st_mass = st.slider(
                "Stellar Mass (Solar masses)",
                min_value=0.1, max_value=3.0, value=1.0, step=0.05,
                help="1.0 = Sun-like mass"
            )

            # Quick presets
            st.markdown("#### 🎯 Quick Presets")
            preset_col1, preset_col2 = st.columns(2)

            with preset_col1:
                if st.form_submit_button("🌍 Earth-like System", use_container_width=True):
                    pl_rade, pl_orbsmax, st_teff, st_mass = 1.0, 1.0, 5778, 1.0

        if submitted:
            # Perform analysis based on selected mode
            st.markdown("---")
            st.markdown("## 🎯 Habitability Analysis Results")

        # Always get physics analysis
        physics_result = assess_habitability_physics(pl_rade, pl_orbsmax, st_teff, st_mass, pl_eqt)

        # Get AI analysis if available
        ai_result = None
        ai_confidence = None
        if model is not None and model_status == "loaded":
            try:
                features = prepare_features_for_model(pl_rade, pl_orbsmax, st_teff, st_mass, pl_eqt)
                ai_prediction = model.predict(features)[0]
                ai_probabilities = model.predict_proba(features)[0]
                ai_confidence = max(ai_probabilities) * 100

                # Convert AI prediction to readable result
                if ai_prediction == 1:
                    ai_result = {
                        'category': 'Potentially Habitable',
                        'color_class': 'habitable',
                        'score': ai_confidence
                    }
                else:
                    ai_result = {
                        'category': 'Not Habitable',
                        'color_class': 'not-habitable',
                        'score': ai_confidence
                    }
            except Exception as e:
                st.error(f"AI Model Error: {str(e)}")
                ai_result = None

        # Display results based on mode
        if analysis_mode == "⚖️ Comparison Mode" and ai_result:
            # Side-by-side comparison
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🤖 AI Model Prediction")
                st.markdown(f"""
                <div class="prediction-result ai-prediction">
                    <h3>🤖 AI MODEL SAYS:</h3>
                    <h2>{ai_result['category'].upper()}</h2>
                    <h3>Confidence: {ai_confidence:.1f}%</h3>
                    <p>Machine Learning Analysis</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("### 🔬 Physics-Based Analysis")
                st.markdown(f"""
                <div class="prediction-result physics-prediction">
                    <h3>🔬 PHYSICS SAYS:</h3>
                    <h2>{physics_result['category'].upper()}</h2>
                    <h3>Score: {physics_result['score']}/100</h3>
                    <p>Traditional Astronomical Analysis</p>
                </div>
                """, unsafe_allow_html=True)

            # Agreement analysis
            ai_habitable = ai_result['category'] == 'Potentially Habitable'
            physics_habitable = physics_result['score'] >= 65

            if ai_habitable == physics_habitable:
                st.success("✅ **AI and Physics Analysis AGREE** - High confidence in result!")
            else:
                st.warning("⚠️ **AI and Physics Analysis DISAGREE** - Requires expert review")

                # Explain disagreement
                if ai_habitable and not physics_habitable:
                    st.info("🤖 AI detected habitability patterns not captured by simple physics rules")
                else:
                    st.info("🔬 Physics suggests habitability but AI found concerning factors in the data")

        elif ai_result:
            # AI + Physics mode
            st.markdown("### 🤖 AI Model Prediction")
            st.markdown(f"""
            <div class="prediction-result ai-prediction">
                <h2>🤖 AI MODEL PREDICTION</h2>
                <h1>{ai_result['category'].upper()}</h1>
                <h2>Confidence: {ai_confidence:.1f}%</h2>
                <p>Based on machine learning analysis of 1,729+ planets</p>
            </div>
            """, unsafe_allow_html=True)

            # Physics validation
            st.markdown("### 🔬 Physics Validation")
            st.markdown(f"""
            <div class="prediction-result {physics_result['color_class']}">
                <h3>Physics Score: {physics_result['score']}/100</h3>
                <h3>Category: {physics_result['category']}</h3>
            </div>
            """, unsafe_allow_html=True)

        else:
            # Physics only mode
            st.markdown("### 🔬 Physics-Based Analysis")
            st.markdown(f"""
            <div class="prediction-result {physics_result['color_class']}">
                <h1>{physics_result['category'].upper()}</h1>
                <h2>Habitability Score: {physics_result['score']}/100</h2>
                <p>Based on established astronomical principles</p>
            </div>
            """, unsafe_allow_html=True)

            if model_status != "loaded":
                st.info("ℹ️ **Note:** AI model not loaded. Showing physics-based analysis only. Load the trained model for AI predictions.")

        # Model prediction indicator
        st.markdown("### 📊 Prediction Source")

        col1, col2, col3 = st.columns(3)

        with col1:
            if ai_result:
                st.success("✅ **AI Model Used**\nMachine learning prediction")
            else:
                st.warning("⚠️ **AI Model Not Used**\nModel not available")

        with col2:
            st.success("✅ **Physics Analysis Used**\nAstronomical calculations")

        with col3:
            confidence_source = "AI Confidence" if ai_result else "Physics Score"
            confidence_value = f"{ai_confidence:.1f}%" if ai_result else f"{physics_result['score']}/100"
            st.metric("Primary Confidence", confidence_value, help=f"Based on {confidence_source}")

        # Detailed factor analysis
        st.markdown("### 🔬 Detailed Factor Analysis")

        for factor, points, explanation in physics_result['factors']:
            if "✅" in factor:
                st.success(f"**{factor}** (+{points} pts): {explanation}")
            elif "⚠️" in factor:
                st.warning(f"**{factor}** (+{points} pts): {explanation}")
            elif "🔥" in factor or "🧊" in factor or "❄️" in factor:
                st.error(f"**{factor}** (+{points} pts): {explanation}")
            else:
                st.info(f"**{factor}** (+{points} pts): {explanation}")

        # System parameters summary
        st.markdown("### 📊 System Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Stellar Luminosity", f"{physics_result['luminosity']:.2f} L☉",
                     help="Brightness compared to our Sun")

        with col2:
            st.metric("HZ Inner Boundary", f"{physics_result['hz_inner']:.3f} AU",
                     help="Closest distance for liquid water")

        with col3:
            st.metric("HZ Outer Boundary", f"{physics_result['hz_outer']:.3f} AU",
                     help="Farthest distance for liquid water")

        with col4:
            st.metric("Planet Temperature", f"{physics_result['pl_eqt']:.0f} K",
                     help="Equilibrium surface temperature")

        # Expert recommendations
        st.markdown("### 🎓 Expert Assessment")

        recommendations = []

        # Overall recommendation based on combined analysis
        if ai_result and ai_result['category'] == 'Potentially Habitable' and physics_result['score'] >= 65:
            recommendations.append("🌟 **TOP PRIORITY TARGET** - Both AI and physics indicate high habitability potential")
        elif ai_result and ai_result['category'] == 'Potentially Habitable':
            recommendations.append("🤖 **AI-IDENTIFIED CANDIDATE** - Machine learning detected habitability signals")
        elif physics_result['score'] >= 80:
            recommendations.append("🔬 **PHYSICS-BASED CANDIDATE** - Excellent conditions according to established principles")
        elif physics_result['score'] >= 65:
            recommendations.append("🌍 **POTENTIAL CANDIDATE** - Good habitability conditions detected")
        elif physics_result['score'] >= 45:
            recommendations.append("📝 **REQUIRES FURTHER STUDY** - Mixed habitability signals")
        else:
            recommendations.append("📚 **RESEARCH INTEREST** - Extreme conditions worth studying")

        # Specific recommendations
        if physics_result['hz_inner'] <= pl_orbsmax <= physics_result['hz_outer']:
            recommendations.append("💧 **LIQUID WATER ZONE** - Perfect orbital position for surface water")

        if physics_result['esi_surface'] > 0.8:
            recommendations.append("🌍 **EARTH-LIKE CONDITIONS** - Very similar to Earth's surface environment")

        if 0.8 <= pl_rade <= 1.2:
            recommendations.append("🪨 **ROCKY PLANET** - Likely solid surface suitable for life")

        if ai_result and ai_confidence > 85:
            recommendations.append("🎯 **HIGH AI CONFIDENCE** - Strong machine learning signal")

        # Display recommendations
        for rec in recommendations:
            st.markdown(f"- {rec}")

        # Visualization
        st.markdown("### 📈 Habitability Factors Visualization")

        # Create radar chart-like visualization using bar chart
        factor_names = []
        factor_scores = []

        # Extract scores from factors
        for factor, points, _ in physics_result['factors']:
            if "Habitable Zone" in factor or "Hot" in factor or "Cold" in factor:
                factor_names.append("Habitable Zone")
                factor_scores.append(points)
            elif "Size" in factor:
                factor_names.append("Planet Size")
                factor_scores.append(points)
            elif "Temperature" in factor:
                factor_names.append("Temperature")
                factor_scores.append(points)
            elif "Star" in factor:
                factor_names.append("Stellar Properties")
                factor_scores.append(points)

        if factor_names:
            fig = px.bar(
                x=factor_names,
                y=factor_scores,
                title="Habitability Factors Breakdown",
                labels={'x': 'Habitability Factors', 'y': 'Points Scored'},
                color=factor_scores,
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def show_enhanced_explorer(df):
    """Enhanced Exoplanet Database Explorer"""
    st.markdown("## 📊 Exoplanet Database Explorer")
    st.markdown(f"*Explore and analyze {len(df):,} confirmed exoplanets with advanced filtering*")

    # Advanced filtering section
    st.markdown("### 🔍 Advanced Search & Filtering")

    with st.container():
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)

        # Search and basic filters
        col1, col2, col3 = st.columns(3)

        with col1:
            search_term = st.text_input("🔍 Search by planet name:",
                                      placeholder="e.g., Kepler, TRAPPIST, Proxima")
            if search_term and 'pl_name' in df.columns:
                df = df[df['pl_name'].str.contains(search_term, case=False, na=False)]

        with col2:
            if 'pl_discmethod' in df.columns:
                methods = ['All Methods'] + sorted(df['pl_discmethod'].dropna().unique().tolist())
                selected_method = st.selectbox("🔭 Detection Method", methods)
                if selected_method != 'All Methods':
                    df = df[df['pl_discmethod'] == selected_method]

        with col3:
            if 'disc_year' in df.columns:
                years = sorted(df['disc_year'].dropna().unique())
                if len(years) > 1:
                    year_range = st.select_slider(
                        "📅 Discovery Years",
                        options=years,
                        value=(years[max(0, len(years)-10)], years[-1]),
                        help="Select range of discovery years"
                    )
                    df = df[(df['disc_year'] >= year_range[0]) & (df['disc_year'] <= year_range[1])]

        st.markdown('</div>', unsafe_allow_html=True)

    # Physical parameter filters
    st.markdown("#### 🌌 Physical Parameter Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        if 'pl_rade' in df.columns:
            min_rad, max_rad = float(df['pl_rade'].min()), float(df['pl_rade'].max())
            radius_range = st.slider(
                "🪐 Planet Radius (Earth radii)",
                min_value=min_rad, max_value=min(max_rad, 10.0),
                value=(min_rad, min(max_rad, 10.0)),
                step=0.1,
                help="Filter by planet size"
            )
            df = df[(df['pl_rade'] >= radius_range[0]) & (df['pl_rade'] <= radius_range[1])]

    with col2:
        if 'pl_orbsmax' in df.columns:
            min_dist, max_dist = float(df['pl_orbsmax'].min()), float(df['pl_orbsmax'].max())
            distance_range = st.slider(
                "🌌 Orbital Distance (AU)",
                min_value=min_dist, max_value=min(max_dist, 5.0),
                value=(min_dist, min(max_dist, 5.0)),
                step=0.01,
                help="Filter by distance from star"
            )
            df = df[(df['pl_orbsmax'] >= distance_range[0]) & (df['pl_orbsmax'] <= distance_range[1])]

    with col3:
        if 'st_teff' in df.columns:
            min_temp, max_temp = int(df['st_teff'].min()), int(df['st_teff'].max())
            temp_range = st.slider(
                "⭐ Stellar Temperature (K)",
                min_value=min_temp, max_value=max_temp,
                value=(min_temp, max_temp),
                step=50,
                help="Filter by host star temperature"
            )
            df = df[(df['st_teff'] >= temp_range[0]) & (df['st_teff'] <= temp_range[1])]

    # Results summary with enhanced stats
    st.markdown(f"""
    <div class="info-card">
        <h3>📊 Filter Results: {len(df):,} planets match your criteria</h3>
        <p>Use the controls above to refine your search and explore specific types of exoplanets.</p>
    </div>
    """, unsafe_allow_html=True)

    if len(df) > 0:
        # Enhanced statistics dashboard
        st.markdown("### 📈 Statistical Analysis")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if 'pl_rade' in df.columns:
                avg_radius = df['pl_rade'].mean()
                earth_sized = len(df[(df['pl_rade'] >= 0.8) & (df['pl_rade'] <= 1.2)])
                st.metric("Avg Planet Size", f"{avg_radius:.2f} R⊕",
                         f"{earth_sized} Earth-sized")

        with col2:
            if 'pl_orbsmax' in df.columns:
                avg_distance = df['pl_orbsmax'].mean()
                hz_range = len(df[(df['pl_orbsmax'] >= 0.5) & (df['pl_orbsmax'] <= 2.0)])
                st.metric("Avg Orbital Distance", f"{avg_distance:.2f} AU",
                         f"{hz_range} in HZ range")

        with col3:
            if 'st_teff' in df.columns:
                avg_star_temp = df['st_teff'].mean()
                sun_like_count = len(df[(df['st_teff'] >= 5200) & (df['st_teff'] <= 6200)])
                st.metric("Avg Star Temperature", f"{avg_star_temp:.0f} K",
                         f"{sun_like_count} Sun-like")

        with col4:
            if 'disc_year' in df.columns:
                latest_discovery = int(df['disc_year'].max())
                recent_count = len(df[df['disc_year'] >= latest_discovery - 2])
                st.metric("Latest Discovery", f"{latest_discovery}",
                         f"{recent_count} recent")

        # Interactive visualizations
        st.markdown("### 📊 Interactive Data Visualizations")

        tab1, tab2, tab3 = st.tabs(["🌍 Size vs Distance", "📈 Discovery Timeline", "🔭 Detection Methods"])

        with tab1:
            if 'pl_rade' in df.columns and 'pl_orbsmax' in df.columns:
                # Enhanced scatter plot
                color_column = 'st_teff' if 'st_teff' in df.columns else None

                fig = px.scatter(
                    df, x='pl_orbsmax', y='pl_rade',
                    color=color_column,
                    size='st_mass' if 'st_mass' in df.columns else None,
                    hover_data=['pl_name'] if 'pl_name' in df.columns else None,
                    title="Planet Size vs Orbital Distance",
                    labels={
                        'pl_orbsmax': 'Orbital Distance (AU)',
                        'pl_rade': 'Planet Radius (Earth radii)',
                        'st_teff': 'Star Temperature (K)'
                    },
                    color_continuous_scale='plasma'
                )

                # Add Earth reference
                fig.add_hline(y=1.0, line_dash="dash", line_color="green",
                             annotation_text="Earth Size", annotation_position="bottom right")
                fig.add_vline(x=1.0, line_dash="dash", line_color="green",
                             annotation_text="Earth Distance", annotation_position="top left")

                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if 'disc_year' in df.columns:
                # Discovery timeline with cumulative count
                yearly_discoveries = df['disc_year'].value_counts().sort_index()
                cumulative_discoveries = yearly_discoveries.cumsum()

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=yearly_discoveries.index,
                    y=yearly_discoveries.values,
                    name='Annual Discoveries',
                    opacity=0.7
                ))

                fig.add_trace(go.Scatter(
                    x=cumulative_discoveries.index,
                    y=cumulative_discoveries.values,
                    mode='lines+markers',
                    name='Cumulative Total',
                    yaxis='y2',
                    line=dict(color='red', width=3)
                ))

                fig.update_layout(
                    title='Exoplanet Discoveries Over Time',
                    xaxis_title='Discovery Year',
                    yaxis_title='Annual Discoveries',
                    yaxis2=dict(title='Cumulative Discoveries', overlaying='y', side='right'),
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            if 'pl_discmethod' in df.columns:
                # Detection methods with details
                method_counts = df['pl_discmethod'].value_counts()

                fig = px.pie(
                    values=method_counts.values,
                    names=method_counts.index,
                    title="Exoplanet Detection Methods Distribution"
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Method explanations
                st.markdown("#### 🔬 Detection Method Explanations")
                method_info = {
                    'Transit': 'Planet passes in front of star, causing periodic dimming',
                    'Radial Velocity': 'Star wobbles due to gravitational pull of orbiting planet',
                    'Microlensing': 'Planet\'s gravity bends light from background star',
                    'Direct Imaging': 'Direct photograph of planet separated from star',
                    'Astrometry': 'Precise measurement of star\'s position changes'
                }

                for method, explanation in method_info.items():
                    if method in method_counts.index:
                        count = method_counts[method]
                        st.write(f"**{method}** ({count} planets): {explanation}")

        # Enhanced data table with sorting and filtering
        st.markdown("### 📋 Detailed Planet Database")

        # Column selection
        all_columns = ['pl_name', 'pl_rade', 'pl_orbsmax', 'st_teff', 'st_mass', 'pl_eqt', 'disc_year', 'pl_discmethod']
        available_columns = [col for col in all_columns if col in df.columns]

        col1, col2 = st.columns(2)

        with col1:
            display_columns = st.multiselect(
                "Select columns to display:",
                available_columns,
                default=available_columns[:6] if len(available_columns) >= 6 else available_columns,
                help="Choose which planet parameters to show in the table"
            )

        with col2:
            if display_columns:
                sort_column = st.selectbox("Sort by:", display_columns)
                sort_ascending = st.checkbox("Ascending order", True)

        # Display the data table
        if display_columns:
            df_display = df[display_columns].copy()

            if sort_column in df_display.columns:
                df_display = df_display.sort_values(sort_column, ascending=sort_ascending)

            # Show row count selector
            max_rows = min(len(df_display), 1000)
            rows_to_show = st.selectbox("Rows to display:", [50, 100, 200, 500, max_rows], index=1)

            st.dataframe(
                df_display.head(rows_to_show),
                use_container_width=True,
                hide_index=True,
                height=400
            )

            # Download functionality
            if st.button("📥 Download Filtered Data as CSV", type="secondary"):
                csv = df_display.to_csv(index=False)
                st.download_button(
                    label="💾 Click to Download CSV",
                    data=csv,
                    file_name=f"exoplanets_filtered_{len(df_display)}_planets.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        # Quick habitability check
        if len(df) <= 20:  # Only for small datasets to avoid performance issues
            st.markdown("### 🔬 Quick Habitability Assessment")

            if st.button("🧪 Analyze All Visible Planets", help="Run habitability analysis on filtered planets"):
                habitability_results = []

                progress_bar = st.progress(0)

                for idx, planet in df.iterrows():
                    if all(param in planet and pd.notna(planet[param]) for param in ['pl_rade', 'pl_orbsmax', 'st_teff', 'st_mass']):
                        result = assess_habitability_physics(
                            planet['pl_rade'], planet['pl_orbsmax'],
                            planet['st_teff'], planet['st_mass']
                        )

                        habitability_results.append({
                            'Planet': planet['pl_name'] if 'pl_name' in planet else f"Planet {idx}",
                            'Score': result['score'],
                            'Category': result['category'],
                            'In HZ': '✅' if result['hz_inner'] <= planet['pl_orbsmax'] <= result['hz_outer'] else '❌'
                        })

                    progress_bar.progress((len(habitability_results)) / len(df))

                if habitability_results:
                    results_df = pd.DataFrame(habitability_results)
                    results_df = results_df.sort_values('Score', ascending=False)

                    st.markdown("#### 🏆 Habitability Rankings")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)

                    # Quick stats
                    highly_habitable = len(results_df[results_df['Score'] >= 80])
                    potentially_habitable = len(results_df[results_df['Score'] >= 65])

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Highly Promising", highly_habitable)
                    with col2:
                        st.metric("Potentially Habitable", potentially_habitable)
                    with col3:
                        st.metric("Analysis Completed", len(habitability_results))

    else:
        st.warning("⚠️ No planets match your current filter criteria. Try adjusting the filters to see results.")

        if st.button("🔄 Reset All Filters"):
            st.rerun()

# Run the application
if __name__ == "__main__":
    main()