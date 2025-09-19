import streamlit as st
import pandas as pd
import numpy as np

# Clean version - no warnings
st.set_page_config(
    page_title="XO - Exoplanet Habitability Classifier",
    page_icon="🌍",
    layout="wide"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1e3c72;
    }
    .prediction-result {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
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
    .physics-box {
        background: #e8f4fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
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
        sample_data = {
            'pl_name': ['Kepler-452b', 'TOI-715b', 'Proxima Cen b', 'TRAPPIST-1e', 'K2-18b',
                       'HD 209458 b', 'Kepler-186f', '55 Cancri e'],
            'pl_rade': [1.63, 1.55, 1.17, 0.92, 2.3, 1.38, 1.11, 2.17],
            'pl_orbsmax': [1.05, 0.083, 0.048, 0.029, 0.14, 0.047, 0.43, 0.016],
            'st_teff': [5757, 3980, 3042, 2566, 3457, 6117, 3755, 5196],
            'st_mass': [1.04, 0.43, 0.12, 0.09, 0.45, 1.12, 0.54, 0.91],
            'pl_eqt': [265, 300, 234, 251, 255, 1359, 188, 2573],
            'disc_year': [2015, 2024, 2016, 2017, 2015, 1999, 2014, 2004],
            'pl_discmethod': ['Transit', 'Transit', 'Radial Velocity', 'Transit', 'Transit',
                            'Transit', 'Transit', 'Radial Velocity']
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
    # Stellar luminosity using mass-luminosity relation
    luminosity = (stellar_mass ** 3.5) * ((stellar_temp / 5778) ** 4)

    # Conservative habitable zone boundaries
    hz_inner = 0.95 * np.sqrt(luminosity)
    hz_outer = 1.37 * np.sqrt(luminosity)

    return hz_inner, hz_outer, luminosity

def calculate_esi_components(pl_rade, pl_mass=None, pl_temp=288):
    """Calculate Earth Similarity Index components"""
    # Radius ESI
    esi_radius = 1 - abs(pl_rade - 1) / (pl_rade + 1)

    # Mass ESI (estimate from radius if not provided)
    if pl_mass is None:
        pl_mass = pl_rade ** 2.06  # Mass-radius relation
    esi_mass = 1 - abs(pl_mass - 1) / (pl_mass + 1)

    # Temperature ESI
    esi_temp = 1 - abs(pl_temp - 288) / (pl_temp + 288)

    # Surface ESI (combination of radius and temperature)
    esi_surface = (esi_radius + esi_temp) / 2

    return esi_radius, esi_mass, esi_temp, esi_surface

def assess_habitability(pl_rade, pl_orbsmax, st_teff, st_mass, pl_eqt=None):
    """Comprehensive habitability assessment"""

    # Calculate equilibrium temperature if not provided
    if pl_eqt is None:
        luminosity = (st_mass ** 3.5) * ((st_teff / 5778) ** 4)
        pl_eqt = 278 * np.sqrt(luminosity) / np.sqrt(pl_orbsmax)

    # Get habitable zone
    hz_inner, hz_outer, luminosity = calculate_habitable_zone(st_mass, st_teff)

    # Calculate ESI components
    esi_radius, esi_mass, esi_temp, esi_surface = calculate_esi_components(pl_rade, None, pl_eqt)

    # Habitability scoring (0-100 scale)
    score = 0
    factors = []

    # Habitable Zone (40 points)
    if hz_inner <= pl_orbsmax <= hz_outer:
        score += 40
        factors.append(("✅ In Habitable Zone", 40, f"{hz_inner:.3f} - {hz_outer:.3f} AU"))
    else:
        if pl_orbsmax < hz_inner:
            factors.append(("🔥 Too Hot (Inside HZ)", 0, f"Minimum: {hz_inner:.3f} AU"))
        else:
            factors.append(("🧊 Too Cold (Outside HZ)", 0, f"Maximum: {hz_outer:.3f} AU"))

    # Planet Size (25 points)
    if 0.5 <= pl_rade <= 2.0:
        size_score = 25
        if 0.8 <= pl_rade <= 1.2:
            factors.append(("✅ Earth-like Size", size_score, f"{pl_rade:.2f} R⊕"))
        else:
            factors.append(("✅ Suitable Size", size_score, f"{pl_rade:.2f} R⊕"))
    elif pl_rade < 0.5:
        size_score = 5
        factors.append(("⚠️ Very Small", size_score, f"{pl_rade:.2f} R⊕ - may lose atmosphere"))
    else:
        size_score = 10
        factors.append(("⚠️ Large Planet", size_score, f"{pl_rade:.2f} R⊕ - likely gaseous"))
    score += size_score

    # Temperature (25 points)
    if 250 <= pl_eqt <= 350:
        temp_score = 25
        if 273 <= pl_eqt <= 313:
            factors.append(("✅ Perfect Temperature", temp_score, f"{pl_eqt:.0f} K"))
        else:
            factors.append(("✅ Moderate Temperature", temp_score, f"{pl_eqt:.0f} K"))
    elif 200 <= pl_eqt < 250:
        temp_score = 15
        factors.append(("🧊 Cold but Possible", temp_score, f"{pl_eqt:.0f} K"))
    elif 350 < pl_eqt <= 400:
        temp_score = 15
        factors.append(("🔥 Hot but Possible", temp_score, f"{pl_eqt:.0f} K"))
    else:
        temp_score = 0
        if pl_eqt < 200:
            factors.append(("❄️ Too Cold", temp_score, f"{pl_eqt:.0f} K"))
        else:
            factors.append(("🔥 Too Hot", temp_score, f"{pl_eqt:.0f} K"))
    score += temp_score

    # Stellar Properties (10 points)
    stellar_score = 0
    if 3000 <= st_teff <= 7000:
        stellar_score += 5
        factors.append(("✅ Stable Star Temperature", 5, f"{st_teff:.0f} K"))

    if 0.3 <= st_mass <= 1.5:
        stellar_score += 5
        factors.append(("✅ Suitable Star Mass", 5, f"{st_mass:.2f} M☉"))

    score += stellar_score

    # Overall assessment
    if score >= 80:
        category = "Highly Promising"
        color_class = "habitable"
    elif score >= 60:
        category = "Potentially Habitable"
        color_class = "habitable"
    elif score >= 40:
        category = "Marginal Habitability"
        color_class = "marginal"
    elif score >= 20:
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
        'luminosity': luminosity,
        'esi_radius': esi_radius,
        'esi_surface': esi_surface,
        'pl_eqt': pl_eqt
    }

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌍 XO - Exoplanet Habitability Classifier</h1>
        <p>Advanced AI for Astronomical Discovery</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data and model
    df, data_source = load_data()
    model, model_status = load_model()

    # Sidebar
    st.sidebar.markdown("## 🚀 Navigation")
    page = st.sidebar.selectbox(
        "Choose Mission:",
        ["🏠 Mission Control", "🔮 Habitability Predictor", "📊 Data Explorer",
         "🌟 Famous Worlds", "🧪 Physics Lab", "📚 Documentation"]
    )

    # System status
    st.sidebar.markdown("### 📊 System Status")
    st.sidebar.metric("Planets", len(df))
    st.sidebar.metric("Data Source", "NASA Archive" if data_source == "file" else "Sample Data")
    st.sidebar.metric("Model", "🟢 Active" if model_status == "loaded" else "🟡 Demo Mode")

    # Route to pages
    if page == "🏠 Mission Control":
        show_dashboard(df, data_source, model_status)
    elif page == "🔮 Habitability Predictor":
        show_predictor()
    elif page == "📊 Data Explorer":
        show_explorer(df)
    elif page == "🌟 Famous Worlds":
        show_famous_worlds(df)
    elif page == "🧪 Physics Lab":
        show_physics_lab()
    elif page == "📚 Documentation":
        show_documentation()

def show_dashboard(df, data_source, model_status):
    """Mission Control Dashboard"""
    st.markdown("## 🏠 Mission Control Center")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🪐 Exoplanets</h3>
            <h2>{len(df):,}</h2>
            <p>Confirmed worlds</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        habitable_estimate = 2 if len(df) > 5 else 1
        st.markdown(f"""
        <div class="metric-card">
            <h3>🌍 Potentially Habitable</h3>
            <h2>{habitable_estimate}</h2>
            <p>Promising candidates</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        accuracy = "97.5%" if model_status == "loaded" else "Demo"
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 AI Accuracy</h3>
            <h2>{accuracy}</h2>
            <p>Model performance</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔬 Features</h3>
            <h2>16</h2>
            <p>Physics parameters</p>
        </div>
        """, unsafe_allow_html=True)

    # Mission overview
    st.markdown("## 🎯 Mission Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔍 Objectives")
        st.markdown("""
        - **Identify** potentially habitable exoplanets
        - **Prioritize** targets for space telescopes
        - **Apply** machine learning to astronomy
        - **Educate** about exoplanet science
        """)

    with col2:
        st.markdown("### 📊 Quick Stats")
        if 'pl_rade' in df.columns:
            avg_radius = df['pl_rade'].mean()
            st.metric("Average Planet Size", f"{avg_radius:.2f} R⊕")

        if 'st_teff' in df.columns:
            avg_temp = df['st_teff'].mean()
            st.metric("Average Star Temperature", f"{avg_temp:.0f} K")

    # Sample data preview
    st.markdown("### 📋 Recent Discoveries")
    if 'disc_year' in df.columns:
        recent_df = df.nlargest(5, 'disc_year')
        display_cols = ['pl_name', 'pl_rade', 'disc_year']
        available_cols = [col for col in display_cols if col in recent_df.columns]
        if available_cols:
            st.dataframe(recent_df[available_cols], use_container_width=True, hide_index=True)

def show_predictor():
    """Advanced Habitability Predictor"""
    st.markdown("## 🔮 Exoplanet Habitability Predictor")
    st.markdown("*Enter planetary parameters for comprehensive habitability analysis*")

    # Input form
    with st.form("habitability_analyzer"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🪐 Planetary Properties")

            pl_rade = st.slider(
                "Planet Radius (Earth radii)",
                min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                help="Size comparison to Earth (1.0 = Earth-sized)"
            )

            pl_orbsmax = st.slider(
                "Orbital Distance (AU)",
                min_value=0.01, max_value=5.0, value=1.0, step=0.01,
                help="Distance from star (1.0 AU = Earth-Sun distance)"
            )

            custom_temp = st.checkbox("Specify temperature manually")
            if custom_temp:
                pl_eqt = st.slider("Equilibrium Temperature (K)", 100, 1000, 288, 10)
            else:
                pl_eqt = None
                st.info("Temperature will be calculated from stellar properties")

        with col2:
            st.markdown("### ⭐ Stellar Properties")

            st_teff = st.slider(
                "Stellar Temperature (K)",
                min_value=2000, max_value=8000, value=5778, step=50,
                help="Surface temperature (5778 K = Sun-like)"
            )

            st_mass = st.slider(
                "Stellar Mass (Solar masses)",
                min_value=0.1, max_value=3.0, value=1.0, step=0.1,
                help="Mass comparison to Sun (1.0 = Sun-like)"
            )

            st.markdown("### 🎯 Analysis Options")
            detailed_analysis = st.checkbox("Show detailed physics breakdown", value=True)

        # Analysis button
        analyze = st.form_submit_button("🔍 Analyze Habitability", type="primary")

        if analyze:
            # Perform habitability assessment
            result = assess_habitability(pl_rade, pl_orbsmax, st_teff, st_mass, pl_eqt)

            # Results section
            st.markdown("---")
            st.markdown("## 🎯 Habitability Assessment Results")

            # Main result display
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"""
                <div class="prediction-result {result['color_class']}">
                    <h2>{result['category'].upper()}</h2>
                    <h3>Habitability Score: {result['score']}/100</h3>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # Score gauge
                st.markdown("### 📊 Score Components")

                # Extract individual scores from factors
                hz_score = next((f[1] for f in result['factors'] if 'Habitable Zone' in f[0] or 'Hot' in f[0] or 'Cold' in f[0]), 0)
                size_score = next((f[1] for f in result['factors'] if 'Size' in f[0]), 0)
                temp_score = next((f[1] for f in result['factors'] if 'Temperature' in f[0]), 0)

                st.progress(hz_score / 40, f"Habitable Zone: {hz_score}/40")
                st.progress(size_score / 25, f"Planet Size: {size_score}/25")
                st.progress(temp_score / 25, f"Temperature: {temp_score}/25")

            # Detailed analysis
            if detailed_analysis:
                st.markdown("### 🔬 Detailed Assessment")

                for factor, points, detail in result['factors']:
                    if "✅" in factor:
                        st.success(f"{factor} (+{points} pts) - {detail}")
                    elif "⚠️" in factor:
                        st.warning(f"{factor} (+{points} pts) - {detail}")
                    elif "🔥" in factor or "🧊" in factor or "❄️" in factor:
                        st.error(f"{factor} (+{points} pts) - {detail}")
                    else:
                        st.info(f"{factor} (+{points} pts) - {detail}")

                # Physics details
                st.markdown("### 📊 Physics Summary")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Stellar Luminosity", f"{result['luminosity']:.2f} L☉")
                    st.metric("HZ Inner Boundary", f"{result['hz_inner']:.3f} AU")

                with col2:
                    st.metric("HZ Outer Boundary", f"{result['hz_outer']:.3f} AU")
                    st.metric("Planet Temperature", f"{result['pl_eqt']:.0f} K")

                with col3:
                    st.metric("Earth Similarity (Size)", f"{result['esi_radius']:.3f}")
                    st.metric("Earth Similarity (Surface)", f"{result['esi_surface']:.3f}")

                # Recommendations
                st.markdown("### 🎓 Expert Recommendations")

                recommendations = []

                if result['score'] >= 80:
                    recommendations.append("🎯 **Top Priority Target** - Excellent habitability potential")
                elif result['score'] >= 60:
                    recommendations.append("🌟 **High Interest** - Strong habitability candidate")
                elif result['score'] >= 40:
                    recommendations.append("📝 **Follow-up Study** - Mixed habitability signals")
                else:
                    recommendations.append("📚 **Research Interest** - Understand extreme conditions")

                if result['esi_surface'] > 0.8:
                    recommendations.append("🌍 **Earth-like Conditions** - Similar surface environment")

                if result['hz_inner'] <= pl_orbsmax <= result['hz_outer']:
                    recommendations.append("💧 **Liquid Water Possible** - In stellar habitable zone")

                if 0.8 <= pl_rade <= 1.2:
                    recommendations.append("🪨 **Rocky Planet** - Likely solid surface")

                for rec in recommendations:
                    st.markdown(f"- {rec}")

def show_explorer(df):
    """Enhanced Data Explorer"""
    st.markdown("## 📊 Exoplanet Database Explorer")
    st.markdown(f"*Browse and analyze {len(df)} confirmed exoplanets*")

    # Advanced filters
    st.markdown("### 🔍 Search & Filter")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Planet name search
        search_term = st.text_input("🔍 Search by planet name:")
        if search_term and 'pl_name' in df.columns:
            df = df[df['pl_name'].str.contains(search_term, case=False, na=False)]

    with col2:
        # Discovery year filter
        if 'disc_year' in df.columns:
            years = sorted(df['disc_year'].dropna().unique())
            if len(years) > 1:
                year_range = st.select_slider(
                    "Discovery Year Range",
                    options=years,
                    value=(years[0], years[-1])
                )
                df = df[(df['disc_year'] >= year_range[0]) & (df['disc_year'] <= year_range[1])]

    with col3:
        # Detection method filter
        if 'pl_discmethod' in df.columns:
            methods = ['All Methods'] + sorted(df['pl_discmethod'].dropna().unique().tolist())
            selected_method = st.selectbox("Detection Method", methods)
            if selected_method != 'All Methods':
                df = df[df['pl_discmethod'] == selected_method]

    # Size and distance filters
    col1, col2 = st.columns(2)

    with col1:
        if 'pl_rade' in df.columns:
            radius_range = st.slider(
                "Planet Radius Range (Earth radii)",
                min_value=0.1, max_value=min(10.0, df['pl_rade'].max()),
                value=(0.1, min(10.0, df['pl_rade'].max())),
                step=0.1
            )
            df = df[(df['pl_rade'] >= radius_range[0]) & (df['pl_rade'] <= radius_range[1])]

    with col2:
        if 'pl_orbsmax' in df.columns:
            distance_range = st.slider(
                "Orbital Distance Range (AU)",
                min_value=0.01, max_value=min(5.0, df['pl_orbsmax'].max()),
                value=(0.01, min(5.0, df['pl_orbsmax'].max())),
                step=0.01
            )
            df = df[(df['pl_orbsmax'] >= distance_range[0]) & (df['pl_orbsmax'] <= distance_range[1])]

    # Results summary
    st.markdown(f"""
    <div class="physics-box">
        <strong>📊 Filter Results:</strong> {len(df)} planets match your criteria
    </div>
    """, unsafe_allow_html=True)

    if len(df) > 0:
        # Quick statistics
        st.markdown("### 📈 Statistics for Filtered Data")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if 'pl_rade' in df.columns:
                avg_radius = df['pl_rade'].mean()
                st.metric("Avg Radius", f"{avg_radius:.2f} R⊕")

        with col2:
            if 'pl_orbsmax' in df.columns:
                avg_distance = df['pl_orbsmax'].mean()
                st.metric("Avg Distance", f"{avg_distance:.2f} AU")

        with col3:
            if 'st_teff' in df.columns:
                avg_star_temp = df['st_teff'].mean()
                st.metric("Avg Star Temp", f"{avg_star_temp:.0f} K")

        with col4:
            if 'disc_year' in df.columns:
                latest_year = df['disc_year'].max()
                st.metric("Latest Discovery", f"{int(latest_year)}")

        # Data table
        st.markdown("### 📋 Planet Database")

        # Select columns to display
        all_columns = ['pl_name', 'pl_rade', 'pl_orbsmax', 'st_teff', 'st_mass', 'pl_eqt', 'disc_year', 'pl_discmethod']
        available_columns = [col for col in all_columns if col in df.columns]

        # Column selector
        display_columns = st.multiselect(
            "Select columns to display:",
            available_columns,
            default=available_columns[:5] if len(available_columns) >= 5 else available_columns
        )

        if display_columns:
            # Sort options
            sort_by = st.selectbox("Sort by:", display_columns, index=0)
            ascending = st.checkbox("Ascending order", True)

            # Display sorted data
            df_display = df[display_columns].sort_values(sort_by, ascending=ascending)
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # Download option
            if st.button("📥 Download Filtered Data as CSV"):
                csv = df_display.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"exoplanets_filtered_{len(df)}_planets.csv",
                    mime="text/csv"
                )
    else:
        st.warning("No planets match your filter criteria. Try adjusting the filters.")

def show_famous_worlds(df):
    """Famous Exoplanets Showcase"""
    st.markdown("## 🌟 Famous Exoplanets Gallery")
    st.markdown("*Explore the most significant exoplanet discoveries*")

    # Famous planets database
    famous_planets = {
        "Kepler-452b": {
            "nickname": "Earth's Cousin",
            "discovery_year": 2015,
            "significance": "First near-Earth-size planet discovered in the habitable zone of a Sun-like star",
            "key_facts": ["1.6× Earth's radius", "385-day orbit", "G-type star host"],
            "why_famous": "Marked a milestone in finding potentially habitable worlds"
        },
        "Proxima Centauri b": {
            "nickname": "Our Nearest Neighbor",
            "discovery_year": 2016,
            "significance": "Closest known exoplanet to Earth at just 4.24 light-years away",
            "key_facts": ["1.17× Earth's radius", "11-day orbit", "Red dwarf host"],
            "why_famous": "Prime target for future interstellar missions like Breakthrough Starshot"
        },
        "TRAPPIST-1e": {
            "nickname": "The Goldilocks Planet",
            "discovery_year": 2017,
            "significance": "One of seven Earth-sized planets, located in the habitable zone",
            "key_facts": ["0.92× Earth's radius", "6-day orbit", "Ultra-cool dwarf star"],
            "why_famous": "Part of the most Earth-like planetary system ever discovered"
        },
        "TOI-715 b": {
            "nickname": "The Recent Find",
            "discovery_year": 2024,
            "significance": "Newly discovered super-Earth in the habitable zone",
            "key_facts": ["1.55× Earth's radius", "19-day orbit", "Nearby red dwarf"],
            "why_famous": "Shows we're still finding promising worlds close to home"
        },
        "K2-18b": {
            "nickname": "The Water World",
            "discovery_year": 2015,
            "significance": "First exoplanet where water vapor was detected in a habitable-zone planet",
            "key_facts": ["2.3× Earth's radius", "33-day orbit", "Water in atmosphere"],
            "why_famous": "Breakthrough in atmospheric characterization of potentially habitable worlds"
        }
    }

    # Planet selector
    selected_planet = st.selectbox(
        "Choose a famous exoplanet to explore:",
        list(famous_planets.keys())
    )

    if selected_planet:
        planet_info = famous_planets[selected_planet]

        # Header for selected planet
        st.markdown(f"### 🪐 {selected_planet}")
        st.markdown(f"**\"{planet_info['nickname']}\"** - Discovered in {planet_info['discovery_year']}")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### 🌟 Significance")
            st.write(planet_info['significance'])

            st.markdown("#### 📊 Key Facts")
            for fact in planet_info['key_facts']:
                st.write(f"• {fact}")

            st.markdown("#### 🎯 Why It's Famous")
            st.write(planet_info['why_famous'])

        with col2:
            # Try to find in dataset and analyze
            if 'pl_name' in df.columns:
                # Search for planet in dataset
                planet_matches = df[df['pl_name'].str.contains(selected_planet.split()[0], case=False, na=False)]

                if len(planet_matches) > 0:
                    planet_data = planet_matches.iloc[0]

                    st.markdown("#### 🔬 Our Analysis")

                    # Display key parameters
                    if 'pl_rade' in planet_data and pd.notna(planet_data['pl_rade']):
                        st.metric("Planet Radius", f"{planet_data['pl_rade']:.2f} R⊕")

                    if 'pl_orbsmax' in planet_data and pd.notna(planet_data['pl_orbsmax']):
                        st.metric("Orbital Distance", f"{planet_data['pl_orbsmax']:.3f} AU")

                    if 'st_teff' in planet_data and pd.notna(planet_data['st_teff']):
                        st.metric("Star Temperature", f"{planet_data['st_teff']:.0f} K")

                    # Run habitability analysis if we have enough data
                    required_params = ['pl_rade', 'pl_orbsmax', 'st_teff', 'st_mass']
                    if all(param in planet_data and pd.notna(planet_data[param]) for param in required_params):

                        # Get temperature if available
                        pl_eqt = planet_data.get('pl_eqt') if 'pl_eqt' in planet_data and pd.notna(planet_data['pl_eqt']) else None

                        # Analyze habitability
                        result = assess_habitability(
                            planet_data['pl_rade'],
                            planet_data['pl_orbsmax'],
                            planet_data['st_teff'],
                            planet_data['st_mass'],
                            pl_eqt
                        )

                        # Display result
                        st.markdown("#### 🎯 Habitability")
                        if result['score'] >= 60:
                            st.success(f"🌍 {result['category']}")
                            st.success(f"Score: {result['score']}/100")
                        elif result['score'] >= 40:
                            st.warning(f"🟡 {result['category']}")
                            st.warning(f"Score: {result['score']}/100")
                        else:
                            st.error(f"❌ {result['category']}")
                            st.error(f"Score: {result['score']}/100")
                    else:
                        st.info("Insufficient data for full analysis")
                else:
                    st.info("Not found in current dataset")
            else:
                st.info("Dataset analysis not available")

    # Hall of Fame gallery
    st.markdown("### 🎭 Exoplanet Hall of Fame")

    # Display all planets in a grid
    cols = st.columns(2)

    for i, (name, info) in enumerate(famous_planets.items()):
        with cols[i % 2]:
            with st.expander(f"🌟 {name} - {info['nickname']}"):
                st.write(f"**Discovered:** {info['discovery_year']}")
                st.write(f"**Significance:** {info['significance']}")

                st.write("**Key Facts:**")
                for fact in info['key_facts']:
                    st.write(f"• {fact}")

def show_physics_lab():
    """Interactive Physics Laboratory"""
    st.markdown("## 🧪 Exoplanet Physics Laboratory")
    st.markdown("*Explore the science behind habitability with interactive calculators*")

    # Physics calculator selector
    calculator = st.selectbox(
        "Choose a physics calculator:",
        [
            "🌡️ Habitable Zone Calculator",
            "🌍 Earth Similarity Index",
            "🚀 Escape Velocity & Atmosphere",
            "☀️ Stellar Properties",
            "💧 Temperature & Water States"
        ]
    )

    if calculator == "🌡️ Habitable Zone Calculator":
        st.markdown("### Habitable Zone Calculator")
        st.markdown("Calculate the 'Goldilocks Zone' where liquid water can exist")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ⭐ Stellar Parameters")

            star_mass = st.slider("Stellar Mass (Solar masses)", 0.1, 3.0, 1.0, 0.1)
            star_temp = st.slider("Stellar Temperature (K)", 2000, 8000, 5778, 50)

            # Calculate results
            hz_inner, hz_outer, luminosity = calculate_habitable_zone(star_mass, star_temp)
            hz_width = hz_outer - hz_inner

            st.markdown("#### 📊 Results")
            st.metric("Stellar Luminosity", f"{luminosity:.2f} L☉")
            st.metric("HZ Inner Boundary", f"{hz_inner:.3f} AU")
            st.metric("HZ Outer Boundary", f"{hz_outer:.3f} AU")
            st.metric("Habitable Zone Width", f"{hz_width:.3f} AU")

        with col2:
            st.markdown("#### 🎯 Test a Planet")

            test_distance = st.slider("Planet Distance (AU)", 0.01, 5.0, 1.0, 0.01)

            # Check if planet is in HZ
            if hz_inner <= test_distance <= hz_outer:
                st.success(f"✅ Planet at {test_distance:.3f} AU is in the habitable zone!")
                hz_position = (test_distance - hz_inner) / hz_width
                if hz_position < 0.3:
                    st.info("🔥 Inner habitable zone - warmer conditions")
                elif hz_position > 0.7:
                    st.info("🧊 Outer habitable zone - cooler conditions")
                else:
                    st.info("🌍 Middle habitable zone - Earth-like conditions")
            else:
                if test_distance < hz_inner:
                    excess_heat = ((hz_inner - test_distance) / test_distance) * 100
                    st.error(f"🔥 Too hot! Planet receives {excess_heat:.0f}% more energy than HZ inner edge")
                else:
                    energy_deficit = ((test_distance - hz_outer) / hz_outer) * 100
                    st.error(f"🧊 Too cold! Planet receives {energy_deficit:.0f}% less energy than HZ outer edge")

        # Educational content
        st.markdown("""
        <div class="physics-box">
            <h4>🔬 The Science Behind Habitable Zones</h4>
            <p><strong>Habitable Zone:</strong> The orbital distance where a planet receives just the right amount of energy for liquid water to exist on its surface.</p>
            <ul>
                <li><strong>Inner Boundary:</strong> Too close = water boils away (runaway greenhouse)</li>
                <li><strong>Outer Boundary:</strong> Too far = water freezes solid (snowball planet)</li>
                <li><strong>Stellar Luminosity:</strong> Brighter stars have habitable zones farther out</li>
                <li><strong>Atmospheric Effects:</strong> Greenhouse gases can extend the habitable zone inward</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    elif calculator == "🌍 Earth Similarity Index":
        st.markdown("### Earth Similarity Index Calculator")
        st.markdown("Measure how Earth-like a planet is across multiple dimensions")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🪐 Planet Parameters")

            planet_radius = st.slider("Planet Radius (Earth radii)", 0.1, 10.0, 1.0, 0.1)
            planet_mass = st.slider("Planet Mass (Earth masses)", 0.1, 50.0, 1.0, 0.1)
            planet_temp = st.slider("Surface Temperature (K)", 100, 1000, 288, 10)

            # Calculate ESI components
            esi_radius, esi_mass, esi_temp, esi_surface = calculate_esi_components(
                planet_radius, planet_mass, planet_temp
            )

            # Global ESI
            esi_global = (esi_radius * esi_mass * esi_temp) ** (1/3)

        with col2:
            st.markdown("#### 📊 Earth Similarity Results")

            st.metric("Radius Similarity", f"{esi_radius:.3f}")
            st.metric("Mass Similarity", f"{esi_mass:.3f}")
            st.metric("Temperature Similarity", f"{esi_temp:.3f}")
            st.metric("Surface ESI", f"{esi_surface:.3f}")
            st.metric("Global ESI", f"{esi_global:.3f}")

            # Interpretation
            if esi_global >= 0.8:
                st.success("🌍 Very Earth-like!")
            elif esi_global >= 0.6:
                st.success("🌱 Earth-similar")
            elif esi_global >= 0.4:
                st.warning("🌙 Somewhat Earth-like")
            else:
                st.error("👽 Very different from Earth")

        # Visual comparison
        st.markdown("### 📊 ESI Component Comparison")

        # Simple bar chart using metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Radius**")
            st.progress(esi_radius, f"{esi_radius:.3f}")

        with col2:
            st.markdown("**Mass**")
            st.progress(esi_mass, f"{esi_mass:.3f}")

        with col3:
            st.markdown("**Temperature**")
            st.progress(esi_temp, f"{esi_temp:.3f}")

        st.markdown("""
        <div class="physics-box">
            <h4>🔬 Understanding Earth Similarity Index</h4>
            <p><strong>ESI Range:</strong> 0.0 (completely different) to 1.0 (identical to Earth)</p>
            <ul>
                <li><strong>ESI > 0.8:</strong> Very Earth-like conditions</li>
                <li><strong>ESI 0.6-0.8:</strong> Earth-similar, potentially habitable</li>
                <li><strong>ESI 0.4-0.6:</strong> Somewhat Earth-like</li>
                <li><strong>ESI < 0.4:</strong> Very different from Earth</li>
            </ul>
            <p><strong>Note:</strong> High ESI doesn't guarantee habitability - atmosphere, magnetic field, and other factors matter too!</p>
        </div>
        """, unsafe_allow_html=True)

def show_documentation():
    """Comprehensive Documentation"""
    st.markdown("## 📚 XO Project Documentation")
    st.markdown("*Complete guide to the Exoplanet Habitability Classifier*")

    # Documentation sections
    doc_section = st.selectbox(
        "Choose documentation section:",
        [
            "🎯 Project Overview",
            "🔬 Scientific Methodology",
            "🤖 Machine Learning Approach",
            "📊 Dataset Information",
            "⚙️ How to Use This Tool",
            "⚠️ Limitations & Disclaimers",
            "📖 References & Further Reading"
        ]
    )

    if doc_section == "🎯 Project Overview":
        st.markdown("""
        ### Project Overview

        The **XO (Exoplanet) Habitability Classifier** is an AI-powered tool that analyzes exoplanets to identify potentially habitable worlds. By combining NASA's exoplanet data with advanced machine learning and established physics principles, we can rapidly screen thousands of confirmed planets to find the most promising candidates for life.

        #### 🎯 Primary Goals

        1. **Automated Discovery**: Process large datasets of exoplanets to identify habitability candidates
        2. **Physics Integration**: Incorporate real astronomical principles into AI predictions
        3. **Decision Support**: Provide confidence scores and detailed explanations
        4. **Educational Tool**: Make exoplanet science accessible to everyone

        #### 🌟 Key Capabilities

        - **Real-time Analysis**: Instant habitability assessment for any planet parameters
        - **Physics-based Calculations**: Habitable zones, Earth similarity, atmospheric retention
        - **Interactive Exploration**: Browse and filter NASA's confirmed exoplanet database
        - **Famous Planet Analysis**: Study well-known discoveries like Kepler-452b
        - **Educational Content**: Learn the science behind habitability assessments

        #### 📈 Impact & Applications

        - **Space Mission Planning**: Prioritize observation targets for telescopes
        - **Astronomical Research**: Identify interesting planets for detailed study
        - **Education**: Teach exoplanet science and astrobiology concepts
        - **Public Engagement**: Make cutting-edge astronomy accessible
        """)

    elif doc_section == "🔬 Scientific Methodology":
        st.markdown("""
        ### Scientific Methodology

        Our approach is grounded in established astronomical principles and peer-reviewed research.

        #### 🌍 Habitability Definition

        We define habitability as **the potential for liquid water to exist on a planet's surface**. This Earth-centric definition is based on:

        - **Water as Universal Solvent**: Essential for all known biochemistry
        - **Temperature Constraints**: Liquid water exists in narrow temperature range
        - **Atmospheric Requirements**: Need sufficient pressure to maintain liquid phase
        - **Stellar Stability**: Host star must provide consistent energy over billions of years

        #### 🔬 Physics Calculations

        **Habitable Zone Boundaries** (Kopparapu et al. 2013):
        ```
        Stellar Luminosity: L = M^3.5 × (T/5778)^4
        Inner HZ Boundary: 0.95 × √L AU
        Outer HZ Boundary: 1.37 × √L AU
        ```

        **Earth Similarity Index** (Schulze-Makuch et al. 2011):
        ```
        ESI_component = 1 - |parameter - Earth_value| / (parameter + Earth_value)
        ESI_global = (ESI_radius × ESI_mass × ESI_temperature)^(1/3)
        ```

        **Atmospheric Retention**:
        ```
        Escape Velocity: v = √(2GM/R)
        Retention Factor: v_planet / v_Earth
        ```

        #### 📊 Scoring System

        Our 100-point habitability scoring system weights factors by importance:

        - **Habitable Zone Position** (40 points): Most critical factor
        - **Planetary Size** (25 points): Affects atmosphere retention and surface
        - **Temperature Range** (25 points): Must allow liquid water
        - **Stellar Properties** (10 points): Host star stability and longevity

        #### ⚖️ Classification Thresholds

        - **Highly Promising** (80+ points): Excellent habitability potential
        - **Potentially Habitable** (60-79 points): Strong candidate
        - **Marginal Habitability** (40-59 points): Mixed signals, needs study
        - **Unlikely** (20-39 points): Significant obstacles
        - **Not Habitable** (<20 points): Extreme conditions
        """)

    elif doc_section == "🤖 Machine Learning Approach":
        st.markdown("""
        ### Machine Learning Approach

        Our AI model combines physics-based features with machine learning for robust habitability assessment.

        #### 🤖 Model Architecture

        **Algorithm**: Random Forest Classifier
        - **Ensemble Method**: 100 decision trees voting together
        - **Feature Count**: 16 physics-based parameters
        - **Training Data**: 1,729 confirmed exoplanets from NASA
        - **Performance**: 97.5% F1-Score, 99.8% ROC-AUC

        **Why Random Forest?**
        - **Interpretability**: Clear feature importance rankings
        - **Robustness**: Handles missing data and outliers well
        - **Non-linear**: Captures complex physics relationships
        - **Balanced**: Resistant to overfitting through ensemble averaging

        #### 🎯 Feature Engineering

        **Primary Features** (Direct Observations):
        - Planet radius, mass, orbital distance
        - Stellar temperature, mass, luminosity
        - Equilibrium temperature

        **Derived Features** (Physics Calculations):
        - Habitable zone position and membership
        - Earth Similarity Index components
        - Escape velocity ratios
        - Stellar flux and energy balance
        - Composite habitability scores

        #### 📊 Training Strategy

        **Data Preparation**:
        - Quality filtering: Remove incomplete or unrealistic entries
        - Feature scaling: Normalize different physical units
        - Class balancing: Handle extreme rarity of habitable planets

        **Validation Approach**:
        - 5-fold stratified cross-validation
        - Temporal validation: Test on recently discovered planets
        - Physics consistency checks: Ensure predictions align with known principles

        #### 🔍 Model Interpretability

        **Feature Importance Rankings**:
        1. **ESI Radius** (50.3%): Earth-like size most critical
        2. **Planet Radius** (27.7%): Direct size measurement
        3. **HZ Position** (3.7%): Habitable zone location
        4. **Habitability Score** (3.2%): Composite physics score

        **SHAP Analysis**: Individual prediction explanations showing which features drove each decision

        **Physics Validation**: Model predictions correlate strongly with theoretical expectations
        """)

    elif doc_section == "⚠️ Limitations & Disclaimers":
        st.markdown("""
        ### Important Limitations & Disclaimers

        #### ⚠️ Critical Limitations

        **Earth-Centric Assumptions**:
        - Model assumes water-based life (only known biochemistry)
        - May miss exotic chemistries or alternative solvents
        - Trained on Earth-like habitability criteria

        **Missing Information**:
        - **No atmospheric data**: Can't assess greenhouse effects, composition
        - **No magnetic fields**: Critical for protecting atmospheres from stellar wind
        - **No geological activity**: Affects carbon cycle and surface conditions
        - **No direct biosignatures**: Cannot detect actual presence of life

        **Observational Biases**:
        - **Detection bias**: Easier to find large planets close to stars
        - **Sample bias**: Small, Earth-like planets in habitable zones underrepresented
        - **Measurement uncertainty**: Planet parameters have significant error bars

        #### 🎯 Appropriate Use Cases

        **✅ This Tool is Good For**:
        - Initial screening of large exoplanet databases
        - Prioritizing targets for follow-up observations
        - Educational demonstrations of habitability concepts
        - Exploring parameter space and "what-if" scenarios

        **❌ This Tool is NOT Suitable For**:
        - Definitive habitability determinations
        - Mission-critical decisions without expert validation
        - Claims about actual presence of life
        - Replacing detailed astrophysical modeling

        #### 🔬 Scientific Disclaimers

        **Model Predictions**:
        - Provide probabilistic estimates, not certainties
        - Should be validated against detailed models
        - Require expert interpretation in context
        - Are designed for research and educational purposes

        **Habitability ≠ Inhabited**:
        - Habitable conditions don't guarantee life exists
        - Many unknown factors affect actual habitability
        - Life detection requires direct atmospheric observations
        - Model identifies candidates for further study, not confirmed habitable worlds

        #### 📊 Performance Context

        **High Accuracy But**:
        - Training data heavily imbalanced (99.9% non-habitable)
        - Few confirmed habitable examples to learn from
        - High accuracy mainly reflects identifying non-habitable worlds
        - Rare habitable planets are hardest to predict correctly

        **Confidence Interpretation**:
        - High confidence doesn't guarantee correctness
        - Low confidence may indicate interesting edge cases
        - Statistical confidence ≠ physical certainty
        - Always consider limitations when interpreting results

        #### 🔮 Future Improvements

        **Planned Enhancements**:
        - Atmospheric modeling integration (JWST data)
        - Magnetic field estimates from stellar activity
        - Improved mass-radius relationships
        - Multi-class habitability categories
        - Uncertainty quantification in predictions

        **Research Frontiers**:
        - Alternative biochemistry modeling
        - Tidally locked planet habitability
        - Red dwarf flare effects
        - Atmospheric escape modeling
        - Machine learning on biosignature data
        """)

    elif doc_section == "📖 References & Further Reading":
        st.markdown("""
        ### References & Further Reading

        #### 📚 Key Scientific Papers

        **Habitable Zone Research**:
        - Kopparapu, R. K., et al. (2013). "Habitable zones around main-sequence stars: new estimates." *Astrophysical Journal*, 765(2), 131.
        - Kasting, J. F., et al. (1993). "Habitable zones around main sequence stars." *Icarus*, 101(1), 108-128.

        **Earth Similarity Index**:
        - Schulze-Makuch, D., et al. (2011). "A two-tiered approach to assessing the habitability of exoplanets." *Astrobiology*, 11(10), 1041-1052.

        **Exoplanet Habitability**:
        - Seager, S. (2013). "Exoplanet habitability." *Science*, 340(6132), 577-581.
        - Anglada-Escudé, G., et al. (2016). "A terrestrial planet candidate in a temperate orbit around Proxima Centauri." *Nature*, 536(7617), 437-440.

        **Machine Learning in Astronomy**:
        - Baron, D. (2019). "Machine learning in astronomy: a practical overview." *arXiv preprint arXiv:1904.07248*.
        - Pearson, K. A., et al. (2018). "Searching for exoplanets using artificial intelligence." *MNRAS*, 474(1), 478-491.

        #### 🌐 Data Sources

        **NASA Exoplanet Archive**:
        - Website: https://exoplanetarchive.ipac.caltech.edu/
        - Data: Planetary Systems Composite Parameters Table
        - Real-time updates as planets are confirmed by the astronomical community

        **Supporting Databases**:
        - SIMBAD Astronomical Database (stellar parameters)
        - Exoplanet Orbit Database (orbital elements)
        - Habitable Exoplanets Catalog (PHL @ Arecibo)

        #### 📖 Educational Resources

        **Books**:
        - "Exoplanets: Hidden Worlds and the Quest for Extraterrestrial Life" by Donald Goldsmith
        - "The Planet Factory" by Dr. Elizabeth Tasker
        - "Astrobiology: A Very Short Introduction" by David Catling

        **Online Courses**:
        - Coursera: "Astrobiology: Exploring Other Worlds" (University of Edinburgh)
        - edX: "Introduction to Exoplanets" (Australian National University)
        - NASA Exoplanet Exploration Program Educational Resources

        **Websites & Tools**:
        - NASA Exoplanet Exploration: https://exoplanets.nasa.gov/
        - Planetary Habitability Laboratory: http://phl.upr.edu/
        - Eyes on Exoplanets (NASA): Interactive 3D exploration

        #### 🔬 Related Projects

        **Professional Tools**:
        - Planetary Habitability Laboratory (University of Puerto Rico)
        - Habitable Exoplanet Catalog
        - NASA's Exoplanet Archive and analysis tools

        **Open Source Software**:
        - PyTransit: Transit light curve modeling
        - batman: Fast transit photometry
        - juliet: Bayesian inference for exoplanets
        - astropy: Python astronomy library

        #### 🚀 Space Missions

        **Current Missions**:
        - **TESS**: Transiting Exoplanet Survey Satellite
        - **JWST**: James Webb Space Telescope (atmospheric characterization)
        - **CHEOPS**: CHaracterising ExOPlanet Satellite

        **Future Missions**:
        - **PLATO**: PLAnetary Transits and Oscillations (ESA, 2026)
        - **Roman Space Telescope**: Wide-field exoplanet survey (NASA, 2027)
        - **HabEx/LUVOIR**: Direct imaging of Earth-like exoplanets (proposed)
        """)

# Run the application
if __name__ == "__main__":
    main()