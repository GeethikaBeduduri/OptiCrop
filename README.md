# OptiCrop: Smart Agricultural Production Optimization Using Machine Learning

OptiCrop is an intelligent agricultural recommendation system that suggests the most suitable crop to cultivate based on soil chemical parameters and local environmental readings.

The recommendation engine leverages a machine learning model trained on physical agricultural parameters to assist farmers in improving crop yields and practicing sustainable farming.

---

## Key Parameters Analysed

1.  **Nitrogen (N)** (mg/kg in soil)
2.  **Phosphorous (P)** (mg/kg in soil)
3.  **Potassium (K)** (mg/kg in soil)
4.  **Temperature** (°C)
5.  **Humidity** (%)
6.  **Soil pH** (0 - 14 scale)
7.  **Rainfall** (mm)

---

## Project Structure

```
c:/opticrop/
│
├── static/
│   └── css/
│       └── style.css            # Custom CSS style designs (Glassmorphism & animations)
│
├── templates/
│   ├── layout.html              # Base layout with navbar and footer
│   ├── index.html               # Home landing page layout
│   ├── about.html               # Agricultural feature information page
│   └── predict.html             # Parameter entry form & prediction outputs
│
├── Crop_recommendation.csv      # Agricultural dataset (auto-generated if missing)
├── train.py                     # Script to generate synthetic dataset and train the ML model
├── model.pkl                    # Serialized Logistic Regression model binary
├── app.py                       # Flask web application server script
├── requirements.txt             # Python packages requirements list
└── README.md                    # Startup guide & project instructions
```

---

## Installation & Setup

### 1. Pre-requisites
Ensure you have **Python 3.8+** installed on your system.

### 2. Install Required Dependencies
Install the required packages using `pip`:
```bash
pip install -r requirements.txt
```

### 3. Generate Data and Train the Model
Run the model training script. This script will automatically generate the `Crop_recommendation.csv` dataset and train a Scikit-Learn `LogisticRegression` classifier, outputting the serialized `model.pkl` binary:
```bash
python train.py
```

### 4. Launch the Web Application
Start the local Flask development server:
```bash
python app.py
```

### 5. Access the Web Interface
Open your web browser and navigate to:
```text
http://127.0.0.1:5000/
```
From here you can navigate to the "Find Your Crop" page, input agricultural readings, and get recommendations.
