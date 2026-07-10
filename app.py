import os
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model upon application startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        print("Pickle model loaded successfully.")
    else:
        print("Model file (model.pkl) not found. Please run train.py first to create it.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if model is None:
            return render_template('predict.html', prediction_text="Error: Recommendation engine is offline.")
        
        try:
            # Parse form fields from request
            n = float(request.form.get('N', 0))
            p = float(request.form.get('P', 0))
            k = float(request.form.get('K', 0))
            temp = float(request.form.get('temperature', 0))
            humid = float(request.form.get('humidity', 0))
            ph = float(request.form.get('ph', 0))
            rain = float(request.form.get('rainfall', 0))
            
            # Format inputs into array structure for model compatibility (1, 7)
            features = np.array([[n, p, k, temp, humid, ph, rain]])
            
            # Run prediction using Scikit-Learn Logistic Regression model
            prediction = model.predict(features)
            crop_name = prediction[0]
            
            # Format prediction output for presentation
            formatted_crop = crop_name.strip().capitalize()
            return render_template('predict.html', prediction_text=f"Recommended Crop: {formatted_crop}")
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return render_template('predict.html', prediction_text="Error: Could not compute prediction. Please check your inputs.")
            
    return render_template('predict.html')

if __name__ == '__main__':
    # Run application on standard localhost port
    app.run(debug=True, port=5000)
