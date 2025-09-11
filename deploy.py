#!/usr/bin/env python3
"""
XO Project Deployment Script
Prepares and validates the Streamlit application for deployment
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if required file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ Missing {description}: {filepath}")
        return False

def check_requirements():
    """Check all deployment requirements"""
    print("🔍 Checking deployment requirements...")

    required_files = [
        ("app/streamlit_app.py", "Streamlit application"),
        ("data/processed/ml_optimized_dataset.csv", "ML dataset"),
        ("models/champion_random_forest.joblib", "Trained model"),
        ("requirements.txt", "Python dependencies")
    ]

    all_present = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_present = False

    return all_present

def create_directories():
    """Create necessary directories"""
    dirs = ["app", "data/processed", "models", "visualizations"]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing dependencies...")

    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {str(e)}")
        return False

def validate_model():
    """Validate that the model loads correctly"""
    print("\n🤖 Validating model...")

    try:
        import joblib
        model = joblib.load('models/champion_random_forest.joblib')
        print("✅ Model loads successfully")

        # Test prediction with dummy data
        import numpy as np
        dummy_features = np.random.random((1, 16))  # 16 features expected
        prediction = model.predict_proba(dummy_features)
        print(f"✅ Model prediction test successful: {prediction[0][1]:.3f}")
        return True

    except Exception as e:
        print(f"❌ Model validation failed: {str(e)}")
        return False

def validate_dataset():
    """Validate that the dataset loads correctly"""
    print("\n📊 Validating dataset...")

    try:
        import pandas as pd
        df = pd.read_csv('data/processed/ml_optimized_dataset.csv')

        print(f"✅ Dataset loads successfully: {len(df):,} rows, {len(df.columns)} columns")

        # Check for required columns
        required_columns = ['pl_name', 'pl_rade', 'pl_orbsmax', 'st_teff', 'st_mass']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"⚠️ Missing columns: {missing_columns}")
        else:
            print("✅ All required columns present")

        return True

    except Exception as e:
        print(f"❌ Dataset validation failed: {str(e)}")
        return False

def test_streamlit_app():
    """Test that Streamlit app can be imported"""
    print("\n🖥️ Testing Streamlit application...")

    try:
        # Change to app directory temporarily
        original_dir = os.getcwd()
        os.chdir('app')

        # Test import
        sys.path.insert(0, '.')
        import streamlit_app

        print("✅ Streamlit app imports successfully")

        # Restore directory and path
        os.chdir(original_dir)
        sys.path.pop(0)

        return True

    except Exception as e:
        print(f"❌ Streamlit app test failed: {str(e)}")
        os.chdir(original_dir)
        return False

def create_launch_script():
    """Create a launch script for easy deployment"""
    launch_script = """#!/bin/bash
# XO Project Launch Script

echo "Starting XO Exoplanet Habitability Classifier..."
echo "Navigate to: http://localhost:8501"
echo "Press Ctrl+C to stop the application"
echo ""

streamlit run app/streamlit_app.py
"""

    with open('launch.sh', 'w', encoding='utf-8') as f:
        f.write(launch_script)

    # Make executable on Unix systems
    try:
        os.chmod('launch.sh', 0o755)
    except:
        pass  # Windows doesn't need this

    print("Created launch script: launch.sh")

def print_deployment_instructions():
    """Print final deployment instructions"""
    print("\n" + "="*60)
    print("🎉 DEPLOYMENT READY!")
    print("="*60)
    print()
    print("🚀 To launch the application:")
    print("   Option 1: streamlit run app/streamlit_app.py")
    print("   Option 2: ./launch.sh (Unix/Mac)")
    print("   Option 3: Double-click launch.sh (some systems)")
    print()
    print("🌐 The app will open at: http://localhost:8501")
    print()
    print("📝 For deployment to Streamlit Cloud:")
    print("   1. Push this repository to GitHub")
    print("   2. Connect your Streamlit Cloud account")
    print("   3. Deploy directly from the repository")
    print()
    print("⚠️ Important notes:")
    print("   - Ensure all required files are committed to git")
    print("   - Check that data files are under GitHub size limits")
    print("   - Model file should be <100MB for Streamlit Cloud")
    print()

def main():
    """Main deployment preparation function"""
    print("🌍 XO Project - Deployment Preparation")
    print("="*50)

    # Create directories
    create_directories()

    # Check requirements
    if not check_requirements():
        print("\n❌ Missing required files. Please ensure all files are present.")
        return False

    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies.")
        return False

    # Validate components
    validations = [
        validate_dataset(),
        validate_model(),
        test_streamlit_app()
    ]

    if not all(validations):
        print("\n❌ Some validation checks failed.")
        return False

    # Create launch script
    create_launch_script()

    # Print success message
    print_deployment_instructions()

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)