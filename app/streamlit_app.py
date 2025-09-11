import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="XO - Exoplanet Habitability Classifier",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1e3c72;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_data
def load_data():
    """Load the dataset and model"""
    try:
        df = pd.read_csv('data/processed/ml_optimized_dataset.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure ml_optimized_dataset.csv is in data/processed/")
        return None

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = joblib.load('models/champion_random_forest.joblib')
        return model
    except FileNotFoundError:
        st.error("Model not found. Please ensure champion_random_forest.joblib is in models/")
        return None

# Helper functions
def impute_features(features_dict):
    """Apply same imputation strategy as training"""
    feature_order = [
        'pl_rade', 'pl_bmasse', 'pl_orbsmax', 'st_teff', 'st_mass', 'pl_eqt',
        'stellar_luminosity', 'hz_position', 'in_habitable_zone',
        'esi_radius', 'esi_mass', 'esi_temperature', 'esi_surface',
        'escape_velocity_ratio', 'stellar_flux', 'habitability_score'
    ]

    # Default values based on median from training data
    defaults = {
        'pl_rade': 1.0, 'pl_bmasse': 1.0, 'pl_orbsmax': 0.1, 'st_teff': 5500, 'st_mass': 1.0,
        'pl_eqt': 300, 'stellar_luminosity': 1.0, 'hz_position': 1.0, 'in_habitable_zone': 0,
        'esi_radius': 0.5, 'esi_mass': 0.5, 'esi_temperature': 0.5, 'esi_surface': 0.5,
        'escape_velocity_ratio': 1.0, 'stellar_flux': 1.0, 'habitability_score': 2.0
    }

    # Create feature vector
    feature_vector = []
    for feature in feature_order:
        if feature in features_dict and not pd.isna(features_dict[feature]):
            feature_vector.append(features_dict[feature])
        else:
            feature_vector.append(defaults.get(feature, 0))

    return np.array(feature_vector)

def calculate_derived_features(pl_rade, pl_orbsmax, st_teff, st_mass, pl_eqt=None):
    """Calculate derived features for prediction"""
    features = {}

    # Basic features
    features['pl_rade'] = pl_rade
    features['pl_orbsmax'] = pl_orbsmax
    features['st_teff'] = st_teff
    features['st_mass'] = st_mass
    features['pl_eqt'] = pl_eqt if pl_eqt else np.nan

    # Derived features
    features['stellar_luminosity'] = st_mass ** 3.5
    features['hz_inner'] = 0.95 * np.sqrt(features['stellar_luminosity'])
    features['hz_outer'] = 1.37 * np.sqrt(features['stellar_luminosity'])
    features['hz_center'] = (features['hz_inner'] + features['hz_outer']) / 2
    features['hz_position'] = pl_orbsmax / features['hz_center']
    features['in_habitable_zone'] = 1 if features['hz_inner'] <= pl_orbsmax <= features['hz_outer'] else 0

    # ESI calculations
    features['esi_radius'] = 1 - abs(pl_rade - 1.0) / (pl_rade + 1.0)
    features['stellar_flux'] = features['stellar_luminosity'] / (pl_orbsmax ** 2)

    # Simplified habitability score
    size_score = features['esi_radius'] * 3
    hz_score = features['in_habitable_zone'] * 3
    temp_score = 2 if pl_eqt and 200 <= pl_eqt <= 400 else 0
    stellar_score = 2 if 3500 <= st_teff <= 7000 else 0
    features['habitability_score'] = size_score + hz_score + temp_score + stellar_score

    return features

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 XO - Exoplanet Habitability Classifier</h1>', unsafe_allow_html=True)
    st.markdown("**Discover potentially habitable worlds using machine learning and astronomical physics**")

    # Load data and model
    df = load_data()
    model = load_model()

    if df is None or model is None:
        st.stop()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", [
        "🏠 Home",
        "🔮 Predict Habitability",
        "📊 Explore Dataset",
        "📈 Model Performance",
        "🌟 Famous Exoplanets",
        "📚 About & Documentation"
    ])

    # Page routing
    if page == "🏠 Home":
        show_home_page(df, model)
    elif page == "🔮 Predict Habitability":
        show_prediction_page(model)
    elif page == "📊 Explore Dataset":
        show_exploration_page(df)
    elif page == "📈 Model Performance":
        show_performance_page()
    elif page == "🌟 Famous Exoplanets":
        show_famous_planets_page(df, model)
    elif page == "📚 About & Documentation":
        show_documentation_page()

def show_home_page(df, model):
    """Home page with project overview"""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## Project Overview")
        st.write("""
        The XO (eXOplanet) Habitability Classifier uses machine learning to predict which
        exoplanets might be capable of supporting life. Built on real NASA data, this model
        analyzes planetary and stellar characteristics to identify promising targets for
        further astronomical study.
        """)

        st.markdown("### Key Features")
        st.write("""
        - **Physics-based analysis**: Incorporates habitable zone calculations, Earth similarity indices, and stellar characteristics
        - **High accuracy**: 97.5% F1 score and 99.8% ROC-AUC on test data
        - **Real NASA data**: Trained on 1,729 confirmed exoplanets from the NASA Exoplanet Archive
        - **Interpretable results**: Clear confidence scores and feature importance explanations
        """)

        # Model limitations warning
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**⚠️ Important Limitations**")
        st.write("""
        This model is designed as a screening tool and has known limitations:
        - May struggle with extreme edge cases (very close-in planets)
        - Optimized for Earth-sized planets in traditional stellar environments
        - Requires human expert validation for critical decisions
        - Should not be used as the sole basis for observation prioritization
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Quick stats
        st.markdown("### Dataset Statistics")

        total_planets = len(df)
        habitable_planets = df['ml_target'].sum() if 'ml_target' in df.columns else 0
        discovery_years = df['disc_year'].max() - df['disc_year'].min() if 'disc_year' in df.columns else 0

        st.metric("Total Exoplanets", f"{total_planets:,}")
        st.metric("Potentially Habitable", f"{habitable_planets:,}")
        st.metric("Discovery Timespan", f"{discovery_years} years")

        # Quick visualization
        if 'stellar_type' in df.columns:
            stellar_dist = df['stellar_type'].value_counts()
            fig = px.pie(values=stellar_dist.values, names=stellar_dist.index,
                        title="Stellar Types in Dataset")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

def show_prediction_page(model):
    """Interactive prediction interface"""
    st.markdown("## 🔮 Predict Exoplanet Habitability")
    st.write("Enter planetary and stellar parameters to assess habitability potential.")

    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Planetary Properties")
            pl_rade = st.number_input(
                "Planet Radius (Earth radii)",
                min_value=0.1, max_value=20.0, value=1.0, step=0.1,
                help="Earth = 1.0, Jupiter ≈ 11.0"
            )

            pl_orbsmax = st.number_input(
                "Orbital Distance (AU)",
                min_value=0.001, max_value=5.0, value=1.0, step=0.01,
                help="Earth = 1.0 AU, Venus = 0.72 AU, Mars = 1.52 AU"
            )

            pl_eqt = st.number_input(
                "Equilibrium Temperature (K)",
                min_value=0, max_value=3000, value=288, step=10,
                help="Earth ≈ 288K, Venus ≈ 460K, Mars ≈ 210K (optional)"
            )

        with col2:
            st.markdown("### Stellar Properties")
            st_teff = st.number_input(
                "Stellar Temperature (K)",
                min_value=2000, max_value=10000, value=5778, step=100,
                help="Sun = 5,778K, Red dwarfs: 2,500-3,500K"
            )

            st_mass = st.number_input(
                "Stellar Mass (Solar masses)",
                min_value=0.1, max_value=3.0, value=1.0, step=0.1,
                help="Sun = 1.0, Red dwarfs: 0.1-0.5, Massive stars: 1.5-3.0"
            )

            planet_name = st.text_input(
                "Planet Name (optional)",
                placeholder="e.g., Kepler-452 b"
            )

        submitted = st.form_submit_button("🚀 Predict Habitability", type="primary")

    if submitted:
        # Calculate derived features
        features = calculate_derived_features(pl_rade, pl_orbsmax, st_teff, st_mass, pl_eqt)

        # Prepare for model prediction
        feature_vector = impute_features(features)

        # Make prediction
        try:
            confidence = model.predict_proba([feature_vector])[0, 1]
            prediction = confidence >= 0.5

            # Display results
            st.markdown("## Prediction Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                if prediction:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown("### 🟢 Potentially Habitable")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown("### 🔴 Likely Not Habitable")
                    st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.metric("Confidence Score", f"{confidence:.1%}")

                # Confidence interpretation
                if confidence > 0.8:
                    conf_text = "Very High Confidence"
                elif confidence > 0.6:
                    conf_text = "High Confidence"
                elif confidence > 0.4:
                    conf_text = "Moderate Confidence"
                else:
                    conf_text = "Low Confidence"
                st.write(f"**{conf_text}**")

            with col3:
                # Habitability score
                hab_score = features['habitability_score']
                st.metric("Habitability Score", f"{hab_score:.1f}/10")
                st.write("Physics-based composite score")

            # Detailed analysis
            st.markdown("### Detailed Analysis")

            analysis_col1, analysis_col2 = st.columns(2)

            with analysis_col1:
                st.markdown("**Key Factors:**")

                # Earth similarity
                esi_radius = features['esi_radius']
                st.write(f"• Earth Size Similarity: {esi_radius:.3f}")

                # Habitable zone
                in_hz = features['in_habitable_zone']
                hz_status = "Yes" if in_hz else "No"
                st.write(f"• In Classical Habitable Zone: {hz_status}")

                # HZ position
                hz_pos = features['hz_position']
                st.write(f"• HZ Position: {hz_pos:.2f} (1.0 = center)")

                # Temperature assessment
                if pl_eqt:
                    if 273 <= pl_eqt <= 373:
                        temp_status = "Excellent (liquid water range)"
                    elif 200 <= pl_eqt <= 450:
                        temp_status = "Good (extended habitable range)"
                    else:
                        temp_status = "Challenging (extreme temperature)"
                    st.write(f"• Temperature: {temp_status}")

            with analysis_col2:
                st.markdown("**Recommendations:**")

                if prediction and confidence > 0.7:
                    st.write("• Priority target for follow-up observations")
                    st.write("• Consider for atmospheric characterization")
                    st.write("• Suitable for detailed habitability studies")
                elif prediction:
                    st.write("• Moderate interest for further study")
                    st.write("• May warrant inclusion in observation programs")
                    st.write("• Consider alongside other factors")
                else:
                    st.write("• Low priority for habitability studies")
                    st.write("• May be interesting for other research")
                    st.write("• Verify parameters if unexpected result")

                # Warning for edge cases
                if pl_orbsmax < 0.02 and st_teff > 4000:
                    st.warning("⚠️ Very close orbit around hot star - likely a lava world despite other favorable conditions")

                if pl_rade > 4:
                    st.warning("⚠️ Large planet likely to be a gas giant - habitability unlikely")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.write("Please check your input parameters and try again.")

def show_exploration_page(df):
    """Dataset exploration and visualization"""
    st.markdown("## 📊 Explore the Exoplanet Dataset")

    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Planets", f"{len(df):,}")
    with col2:
        if 'ml_target' in df.columns:
            habitable_count = df['ml_target'].sum()
            st.metric("Potentially Habitable", f"{habitable_count:,}")
    with col3:
        if 'disc_year' in df.columns:
            year_range = f"{df['disc_year'].min()}-{df['disc_year'].max()}"
            st.metric("Discovery Years", year_range)
    with col4:
        method_count = df['discoverymethod'].nunique() if 'discoverymethod' in df.columns else 0
        st.metric("Discovery Methods", method_count)

    # Interactive filters
    st.markdown("### Filter and Explore")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        # Planet size filter
        if 'pl_rade' in df.columns:
            size_range = st.slider(
                "Planet Size Range (Earth radii)",
                min_value=float(df['pl_rade'].min()),
                max_value=float(df['pl_rade'].max()),
                value=(0.5, 2.0)
            )

        # Discovery method filter
        if 'discoverymethod' in df.columns:
            methods = st.multiselect(
                "Discovery Methods",
                options=df['discoverymethod'].unique(),
                default=df['discoverymethod'].value_counts().head(3).index.tolist()
            )

    with filter_col2:
        # Stellar temperature filter
        if 'st_teff' in df.columns:
            temp_range = st.slider(
                "Stellar Temperature Range (K)",
                min_value=int(df['st_teff'].min()),
                max_value=int(df['st_teff'].max()),
                value=(3000, 7000)
            )

        # Year filter
        if 'disc_year' in df.columns:
            year_range = st.slider(
                "Discovery Year Range",
                min_value=int(df['disc_year'].min()),
                max_value=int(df['disc_year'].max()),
                value=(2010, int(df['disc_year'].max()))
            )

    # Apply filters
    filtered_df = df.copy()

    if 'pl_rade' in df.columns and size_range:
        filtered_df = filtered_df[
            (filtered_df['pl_rade'] >= size_range[0]) &
            (filtered_df['pl_rade'] <= size_range[1])
        ]

    if 'discoverymethod' in df.columns and methods:
        filtered_df = filtered_df[filtered_df['discoverymethod'].isin(methods)]

    if 'st_teff' in df.columns and temp_range:
        filtered_df = filtered_df[
            (filtered_df['st_teff'] >= temp_range[0]) &
            (filtered_df['st_teff'] <= temp_range[1])
        ]

    if 'disc_year' in df.columns and year_range:
        filtered_df = filtered_df[
            (filtered_df['disc_year'] >= year_range[0]) &
            (filtered_df['disc_year'] <= year_range[1])
        ]

    st.write(f"Showing {len(filtered_df):,} planets after filtering")

    # Visualizations
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        # Planet size vs orbital distance
        if 'pl_rade' in filtered_df.columns and 'pl_orbsmax' in filtered_df.columns:
            fig = px.scatter(
                filtered_df.dropna(subset=['pl_rade', 'pl_orbsmax']),
                x='pl_orbsmax', y='pl_rade',
                color='ml_target' if 'ml_target' in filtered_df.columns else None,
                title="Planet Size vs Orbital Distance",
                labels={'pl_orbsmax': 'Orbital Distance (AU)', 'pl_rade': 'Planet Radius (Earth radii)'},
                hover_data=['pl_name'] if 'pl_name' in filtered_df.columns else None
            )
            fig.update_xaxis(type="log")
            st.plotly_chart(fig, use_container_width=True)

    with viz_col2:
        # Discovery timeline
        if 'disc_year' in filtered_df.columns:
            timeline_data = filtered_df['disc_year'].value_counts().sort_index()
            fig = px.bar(
                x=timeline_data.index, y=timeline_data.values,
                title="Exoplanet Discoveries Over Time",
                labels={'x': 'Year', 'y': 'Number of Discoveries'}
            )
            st.plotly_chart(fig, use_container_width=True)

    # Data table
    st.markdown("### Sample Data")
    display_columns = ['pl_name', 'hostname', 'pl_rade', 'pl_orbsmax', 'st_teff', 'discoverymethod']
    available_columns = [col for col in display_columns if col in filtered_df.columns]

    if available_columns:
        st.dataframe(
            filtered_df[available_columns].head(20),
            use_container_width=True
        )

def show_performance_page():
    """Model performance metrics and analysis"""
    st.markdown("## 📈 Model Performance Analysis")

    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("F1 Score", "97.5%", delta="32.5% above target")
    with col2:
        st.metric("ROC-AUC", "99.8%", delta="14.8% above target")
    with col3:
        st.metric("Precision", "98.8%", delta="Low false positive rate")
    with col4:
        st.metric("Recall", "96.3%", delta="High discovery rate")

    # Model insights
    st.markdown("### Key Model Insights")

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.markdown("**Top Predictive Features:**")
        st.write("1. **ESI Radius** (50.3%) - Earth size similarity")
        st.write("2. **Planet Radius** (27.7%) - Direct size measurement")
        st.write("3. **HZ Position** (3.7%) - Distance from habitable zone center")
        st.write("4. **Habitability Score** (3.2%) - Composite physics score")
        st.write("5. **Orbital Distance** (3.1%) - Distance from host star")

    with insight_col2:
        st.markdown("**Physics Validation:**")
        st.write("✅ Earth-sized planets strongly preferred (96.6% success rate)")
        st.write("✅ Hot Jupiters correctly rejected as non-habitable")
        st.write("✅ Model learned astronomically sound patterns")
        st.write("⚠️ Some edge cases require careful interpretation")
        st.write("⚠️ Conservative bias may miss some habitable worlds")

    # Performance by categories
    st.markdown("### Performance by Planet Categories")

    perf_data = {
        'Category': ['Earth-like (0.8-1.5 R⊕)', 'Super-Earths (1.5-2.5 R⊕)',
                     'Mini-Neptunes (2.5-4 R⊕)', 'Gas Giants (>4 R⊕)'],
        'Habitability Rate': ['96.6%', '3.1%', '2.4%', '0.0%'],
        'Sample Size': [386, 582, 500, 261],
        'Confidence': ['Very High', 'High', 'High', 'Very High']
    }

    st.dataframe(pd.DataFrame(perf_data), use_container_width=True)

    # Limitations and recommendations
    st.markdown("### Model Limitations & Usage Recommendations")

    limitations_col1, limitations_col2 = st.columns(2)

    with limitations_col1:
        st.markdown("**Known Limitations:**")
        st.write("• May misclassify very close-in Earth-sized planets (lava worlds)")
        st.write("• Optimized for planets around main sequence stars")
        st.write("• Limited by training data discovery biases")
        st.write("• Does not consider atmospheric composition")
        st.write("• Simplified tidal locking assessment")

    with limitations_col2:
        st.markdown("**Recommended Usage:**")
        st.write("• Use as initial screening tool for large datasets")
        st.write("• Combine with expert astronomical judgment")
        st.write("• Focus on high-confidence predictions (>80%)")
        st.write("• Validate edge cases with additional analysis")
        st.write("• Consider observation constraints and priorities")

def show_famous_planets_page(df, model):
    """Analysis of famous exoplanets"""
    st.markdown("## 🌟 Famous Exoplanets Analysis")
    st.write("See how our model performs on well-known exoplanets")

    # Famous planets data
    famous_planets = [
        {
            'name': 'Kepler-452 b',
            'category': 'Known Potentially Habitable',
            'pl_rade': 1.6,
            'pl_orbsmax': 1.05,
            'st_teff': 5757,
            'st_mass': 1.04,
            'description': 'Called "Earths cousin" - first near-Earth-size planet in habitable zone of Sun-like star'
        },
        {
            'name': 'Proxima Centauri b',
            'category': 'Known Potentially Habitable',
            'pl_rade': 1.1,
            'pl_orbsmax': 0.05,
            'st_teff': 3042,
            'st_mass': 0.12,
            'description': 'Closest known exoplanet to Earth, orbiting in habitable zone of red dwarf'
        },
        {
            'name': 'TRAPPIST-1 e',
            'category': 'Known Potentially Habitable',
            'pl_rade': 0.91,
            'pl_orbsmax': 0.029,
            'st_teff': 2511,
            'st_mass': 0.09,
            'description': 'One of seven Earth-sized planets in TRAPPIST-1 system, likely in habitable zone'
        },
        {
            'name': 'HD 209458 b',
            'category': 'Famous Non-Habitable',
            'pl_rade': 14.0,
            'pl_orbsmax': 0.047,
            'st_teff': 6065,
            'st_mass': 1.15,
            'description': 'First exoplanet found transiting its star, hot Jupiter with extreme temperatures'
        },
        {
            'name': 'Kepler-78 b',
            'category': 'Edge Case',
            'pl_rade': 1.20,
            'pl_orbsmax': 0.009,
            'st_teff': 5089,
            'st_mass': 0.81,
            'description': 'Earth-sized but extremely close to star - a lava world despite Earth-like size'
        }
    ]

    # Test each famous planet
    results = []

    for planet in famous_planets:
        features = calculate_derived_features(
            planet['pl_rade'], planet['pl_orbsmax'],
            planet['st_teff'], planet['st_mass']
        )

        feature_vector = impute_features(features)
        confidence = model.predict_proba([feature_vector])[0, 1]
        prediction = confidence >= 0.5

        results.append({
            'Planet': planet['name'],
            'Category': planet['category'],
            'Prediction': 'Habitable' if prediction else 'Not Habitable',
            'Confidence': f"{confidence:.1%}",
            'Description': planet['description']
        })

    # Display results
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)

    # Analysis by category
    st.markdown("### Analysis by Category")

    for category in ['Known Potentially Habitable', 'Famous Non-Habitable', 'Edge Case']:
        category_results = [r for r in results if r['Category'] == category]

        if category_results:
            st.markdown(f"**{category}:**")

            for result in category_results:
                status = "🟢" if result['Prediction'] == 'Habitable' else "🔴"
                st.write(f"{status} **{result['Planet']}** - {result['Prediction']} ({result['Confidence']})")
                st.write(f"   {result['Description']}")

            st.write("")

    # Model validation summary
    st.markdown("### Validation Summary")

    habitable_planets = [r for r in results if r['Category'] == 'Known Potentially Habitable']
    non_habitable_planets = [r for r in results if r['Category'] == 'Famous Non-Habitable']