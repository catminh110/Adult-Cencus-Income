"""
Flask Web Application for Adult Census Income Prediction
Features: Prediction, Explanation, Ethics/Limitations
"""
from flask import Flask, render_template, request, jsonify
import sys
import os
import joblib
import numpy as np
import pandas as pd
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

app = Flask(__name__)

# Load model and preprocessor
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.joblib')
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'preprocessor.joblib')
FEATURE_NAMES_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_names.txt')

# Global variables for model and preprocessor
model = None
preprocessor = None
feature_names = None

def load_model():
    """Load the trained model and preprocessor"""
    global model, preprocessor, feature_names
    
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        
        with open(FEATURE_NAMES_PATH, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        print("Model and preprocessor loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Load model on startup
load_model()

# Feature definitions for the form
FEATURE_DEFINITIONS = {
    'age': {
        'label': 'Age',
        'type': 'number',
        'min': 17,
        'max': 90,
        'default': 30,
        'description': 'Age of the person'
    },
    'workclass': {
        'label': 'Workclass',
        'type': 'select',
        'options': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                   'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', 'Unknown'],
        'default': 'Private',
        'description': 'Type of employment'
    },
    'fnlwgt': {
        'label': 'Final Weight',
        'type': 'number',
        'min': 10000,
        'max': 1500000,
        'default': 200000,
        'description': 'Census weight representing population demographics'
    },
    'education': {
        'label': 'Education',
        'type': 'select',
        'options': ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', 
                   '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc', 
                   'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate'],
        'default': 'HS-grad',
        'description': 'Highest education level achieved'
    },
    'education-num': {
        'label': 'Education Number',
        'type': 'number',
        'min': 1,
        'max': 16,
        'default': 10,
        'description': 'Number of years of education'
    },
    'marital-status': {
        'label': 'Marital Status',
        'type': 'select',
        'options': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
                   'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
        'default': 'Never-married',
        'description': 'Current marital status'
    },
    'occupation': {
        'label': 'Occupation',
        'type': 'select',
        'options': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                   'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                   'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                   'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                   'Armed-Forces', 'Unknown'],
        'default': 'Sales',
        'description': 'Type of occupation'
    },
    'relationship': {
        'label': 'Relationship',
        'type': 'select',
        'options': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 
                   'Other-relative', 'Unmarried'],
        'default': 'Not-in-family',
        'description': 'Family relationship role'
    },
    'race': {
        'label': 'Race',
        'type': 'select',
        'options': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 
                   'Other', 'Black'],
        'default': 'White',
        'description': 'Self-reported race'
    },
    'sex': {
        'label': 'Sex',
        'type': 'select',
        'options': ['Male', 'Female'],
        'default': 'Male',
        'description': 'Gender'
    },
    'capital-gain': {
        'label': 'Capital Gain',
        'type': 'number',
        'min': 0,
        'max': 100000,
        'default': 0,
        'description': 'Capital gains income (USD)'
    },
    'capital-loss': {
        'label': 'Capital Loss',
        'type': 'number',
        'min': 0,
        'max': 5000,
        'default': 0,
        'description': 'Capital losses (USD)'
    },
    'hours-per-week': {
        'label': 'Hours per Week',
        'type': 'number',
        'min': 1,
        'max': 99,
        'default': 40,
        'description': 'Average hours worked per week'
    },
    'native-country': {
        'label': 'Native Country',
        'type': 'select',
        'options': ['United-States', 'Cuba', 'Jamaica', 'India', 'Mexico', 'South',
                   'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
                   'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
                   'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic',
                   'El-Salvador', 'France', 'Guatemala', 'China', 'Japan', 'Yemen',
                   'Nicaragua', 'Peru', 'Greece', 'Trinadad&Tobago', 
                   'Outlying-US(Guam-USVI-etc)', 'Hungary', 'Hong', 'Ireland',
                   'Holand-Netherlands', 'Unknown'],
        'default': 'United-States',
        'description': 'Country of birth'
    }
}

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', features=FEATURE_DEFINITIONS)

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction from form data"""
    try:
        # Get form data
        data = {}
        for key in FEATURE_DEFINITIONS.keys():
            if FEATURE_DEFINITIONS[key]['type'] == 'number':
                data[key] = float(request.form.get(key, FEATURE_DEFINITIONS[key]['default']))
            else:
                data[key] = request.form.get(key, FEATURE_DEFINITIONS[key]['default'])
        
        # Create DataFrame
        input_df = pd.DataFrame([data])
        
        # Transform using preprocessor
        X = preprocessor.transform(input_df)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            prediction = 1 if proba[1] > 0.5 else 0
            confidence = proba[1] if prediction == 1 else proba[0]
        else:
            prediction = model.predict(X)[0]
            decision = model.decision_function(X)[0] if hasattr(model, 'decision_function') else 0
            confidence = min(abs(decision) / 3, 1.0)
        
        # Generate explanation
        explanation = generate_explanation(data, prediction, confidence)
        
        # Generate ethics warning
        ethics_warning = generate_ethics_warning()
        
        result = {
            'success': True,
            'prediction': '>50K' if prediction == 1 else '<=50K',
            'confidence': f"{confidence*100:.1f}%",
            'explanation': explanation,
            'ethics_warning': ethics_warning
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def generate_explanation(data, prediction, confidence):
    """Generate human-readable explanation for the prediction"""
    explanation = []
    
    # Base explanation
    if prediction == 1:
        explanation.append(f"The model predicts income **>50K/year** with {confidence*100:.1f}% confidence.")
    else:
        explanation.append(f"The model predicts income **<=50K/year** with {confidence*100:.1f}% confidence.")
    
    # Key factors
    factors = []
    
    # Education factor
    edu_level = data['education-num']
    if edu_level >= 13:
        factors.append(f"High education level ({data['education']}) - positive factor")
    elif edu_level <= 9:
        factors.append(f"Lower education level ({data['education']}) - limiting factor")
    
    # Age factor
    age = data['age']
    if age >= 35 and age <= 55:
        factors.append(f"Age in peak earning years ({int(age)} years)")
    elif age < 25:
        factors.append(f"Young age ({int(age)} years) - typically correlates with lower income")
    
    # Hours per week
    hours = data['hours-per-week']
    if hours > 45:
        factors.append(f"Works many hours ({int(hours)} hours/week)")
    elif hours < 30:
        factors.append(f"Part-time work ({int(hours)} hours/week)")
    
    # Occupation
    high_income_occ = ['Exec-managerial', 'Prof-specialty']
    if data['occupation'] in high_income_occ:
        factors.append(f"High-income occupation ({data['occupation']})")
    
    # Capital gain
    if data['capital-gain'] > 5000:
        factors.append(f"Significant capital gains (${int(data['capital-gain']):,})")
    
    if factors:
        explanation.append("\n**Key influencing factors:**")
        for factor in factors:
            explanation.append(f"- {factor}")
    
    return "\n".join(explanation)

def generate_ethics_warning():
    """Generate ethics and limitations warning"""
    warning = {
        'title': 'Ethics & Limitations Warning',
        'points': [
            {
                'title': 'No Social Inference',
                'content': 'This model is based on statistical data and does not reflect individual worth or potential.'
            },
            {
                'title': 'Bias in Data',
                'content': 'Training data may contain historical biases regarding gender, race, and country of origin.'
            },
            {
                'title': 'Prediction Limitations',
                'content': 'The model only predicts probability based on demographic features, not actual individual capability.'
            },
            {
                'title': 'Not for Employment Decisions',
                'content': 'Results should not be used for hiring, lending, or personal evaluation decisions.'
            },
            {
                'title': 'Temporal Context',
                'content': 'Data from 1994 may not accurately reflect current economic conditions.'
            }
        ]
    }
    return warning

@app.route('/api/features')
def get_features():
    """API endpoint to get feature definitions"""
    return jsonify(FEATURE_DEFINITIONS)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
