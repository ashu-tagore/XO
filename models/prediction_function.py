
def predict_exoplanet_habitability(planet_features):
    """
    Predict exoplanet habitability using trained Random Forest model

    Parameters:
    planet_features (dict): Dictionary with feature values
    Required features: pl_rade, pl_orbsmax, st_teff, st_mass, etc.

    Returns:
    dict: Prediction results with confidence score
    """
    import joblib
    import numpy as np

    # Load model
    model = joblib.load('models/champion_random_forest.joblib')

    # Extract features in correct order
    feature_order = ['pl_rade', 'pl_bmasse', 'pl_orbsmax', 'st_teff', 'st_mass', 'pl_eqt', 'stellar_luminosity', 'hz_position', 'in_habitable_zone', 'esi_radius', 'esi_mass', 'esi_temperature', 'esi_surface', 'escape_velocity_ratio', 'stellar_flux', 'habitability_score']
    features_array = np.array([planet_features.get(f, np.nan) for f in feature_order])

    # Predict
    confidence = model.predict_proba([features_array])[0, 1]
    prediction = confidence >= 0.5

    return {{
        'habitable': bool(prediction),
        'confidence': float(confidence),
        'interpretation': 'Potentially habitable' if prediction else 'Likely not habitable'
    }}
