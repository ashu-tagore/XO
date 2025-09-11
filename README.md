# XO - Exoplanet Habitability Classifier Web Application

## Overview

A professional Streamlit web application for the XO Exoplanet Habitability Classifier. This app provides an interactive interface to explore exoplanet data and predict habitability using a trained machine learning model.

## Features

- **🏠 Home Page**: Project overview and quick statistics
- **🔮 Predict Habitability**: Interactive prediction interface with detailed analysis
- **📊 Explore Dataset**: Data visualization and filtering tools
- **📈 Model Performance**: Comprehensive performance metrics and insights
- **🌟 Famous Exoplanets**: Analysis of well-known exoplanets
- **📚 Documentation**: Complete methodology and usage guidelines

## Setup Instructions

### Prerequisites

Ensure you have the following files in the correct locations:

```
app/
├── streamlit_app.py
├── requirements.txt
└── README.md

data/processed/
└── ml_optimized_dataset.csv

models/
└── champion_random_forest.joblib
```

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

3. **Access the app:**
   Open your browser to `http://localhost:8501`

## Usage Guidelines

### Prediction Interface

1. **Enter planetary parameters:**
   - Planet radius (Earth radii)
   - Orbital distance (AU)
   - Equilibrium temperature (K) - optional

2. **Enter stellar parameters:**
   - Stellar temperature (K)
   - Stellar mass (solar masses)

3. **Get results:**
   - Habitability prediction (Yes/No)
   - Confidence score (0-100%)
   - Detailed physics analysis
   - Expert recommendations

### Important Limitations

⚠️ **This model is a screening tool, not a definitive assessment**

- May struggle with extreme edge cases
- Requires expert validation for critical decisions
- Best used for initial dataset screening
- Should not be sole basis for observation priorities

### Model Performance

- **F1 Score:** 97.5%
- **ROC-AUC:** 99.8%
- **Precision:** 98.8%
- **Recall:** 96.3%

Trained on 1,729 confirmed exoplanets from NASA Exoplanet Archive.

## Technical Details

### Architecture

- **Frontend:** Streamlit with custom CSS styling
- **Visualization:** Plotly for interactive charts
- **Model:** Random Forest Classifier (scikit-learn)
- **Data:** NASA Exoplanet Archive PS table

### Features Used

1. **ESI Radius** (50.3%) - Earth size similarity
2. **Planet Radius** (27.7%) - Direct size measurement
3. **HZ Position** (3.7%) - Habitable zone location
4. **Habitability Score** (3.2%) - Composite physics score

### Physics-Based Calculations

- Habitable zone boundaries using stellar luminosity
- Earth Similarity Index components
- Stellar flux and equilibrium temperature estimates
- Atmospheric retention potential

## Deployment Options

### Local Development

```bash
streamlit run app/streamlit_app.py
```

### Streamlit Cloud

1. Push code to GitHub repository
2. Connect Streamlit Cloud to repository
3. Deploy with automatic updates

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501"]
```

## Troubleshooting

### Common Issues

1. **"Model not found" error:**
   - Ensure `champion_random_forest.joblib` is in `models/` directory
   - Check file permissions

2. **"Dataset not found" error:**
   - Ensure `ml_optimized_dataset.csv` is in `data/processed/` directory
   - Verify CSV format and columns

3. **Import errors:**
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)

### Performance Issues

- Large dataset may cause slow loading
- Consider data sampling for faster development
- Use caching for model loading (`@st.cache_resource`)

## Contributing

When modifying the application:

1. Test all pages and functionality
2. Verify model predictions work correctly
3. Check responsive design on different screen sizes
4. Update documentation as needed

## Contact

For questions about the model or application, refer to the documentation page within the app or consult the project notebooks for detailed methodology.