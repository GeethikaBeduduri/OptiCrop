import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def generate_synthetic_data(filepath):
    print("Crop_recommendation.csv not found. Generating realistic synthetic dataset...")
    
    crops = [
        'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 
        'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 
        'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 
        'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'
    ]
    
    # Baseline physiological ranges for the 22 crops
    # Format: crop: (N_range, P_range, K_range, Temp_range, Humid_range, pH_range, Rain_range)
    profiles = {
        'rice': ((60, 99), (35, 60), (35, 45), (20, 27), (80, 85), (5.5, 7.0), (180, 290)),
        'maize': ((60, 99), (35, 60), (15, 25), (18, 27), (55, 70), (5.5, 7.0), (60, 110)),
        'chickpea': ((20, 59), (55, 80), (75, 85), (17, 21), (15, 20), (5.5, 9.0), (65, 95)),
        'kidneybeans': ((20, 40), (55, 80), (15, 25), (15, 25), (18, 25), (5.5, 6.0), (60, 150)),
        'pigeonpeas': ((0, 40), (55, 80), (15, 25), (18, 36), (30, 69), (4.5, 7.8), (90, 199)),
        'mothbeans': ((0, 40), (35, 60), (15, 25), (24, 31), (40, 65), (3.5, 9.0), (30, 75)),
        'mungbean': ((0, 40), (35, 60), (15, 25), (27, 29), (80, 90), (6.2, 7.2), (36, 60)),
        'blackgram': ((20, 60), (55, 80), (15, 25), (25, 35), (60, 70), (6.5, 7.5), (60, 75)),
        'lentil': ((0, 40), (55, 80), (15, 25), (18, 30), (60, 70), (5.9, 6.9), (35, 55)),
        'pomegranate': ((0, 40), (5, 30), (35, 45), (18, 25), (85, 90), (5.5, 7.2), (100, 110)),
        'banana': ((80, 120), (75, 95), (45, 55), (25, 29), (75, 85), (5.5, 6.5), (90, 115)),
        'mango': ((0, 40), (15, 40), (25, 35), (27, 36), (45, 55), (4.5, 7.0), (89, 101)),
        'grapes': ((20, 40), (120, 145), (195, 205), (8, 42), (80, 84), (5.5, 6.5), (65, 75)),
        'watermelon': ((80, 100), (5, 30), (45, 55), (24, 27), (80, 90), (6.0, 7.0), (40, 60)),
        'muskmelon': ((80, 100), (5, 30), (45, 55), (27, 30), (90, 95), (6.0, 6.8), (20, 30)),
        'apple': ((0, 40), (120, 145), (195, 205), (21, 24), (90, 95), (5.5, 6.5), (100, 125)),
        'orange': ((0, 40), (5, 30), (5, 15), (11, 35), (90, 95), (6.0, 8.0), (100, 120)),
        'papaya': ((30, 70), (45, 70), (45, 55), (23, 44), (90, 95), (6.5, 7.0), (90, 250)),
        'coconut': ((0, 40), (5, 30), (25, 35), (25, 30), (90, 99), (5.5, 6.5), (130, 230)),
        'cotton': ((100, 140), (35, 60), (15, 25), (22, 26), (75, 85), (5.8, 8.0), (60, 100)),
        'jute': ((60, 100), (35, 60), (35, 45), (23, 27), (70, 90), (6.0, 7.0), (150, 200)),
        'coffee': ((80, 120), (15, 40), (25, 35), (23, 28), (50, 69), (6.0, 7.5), (140, 200))
    }
    
    data = []
    np.random.seed(42)
    
    for crop in crops:
        ranges = profiles[crop]
        for _ in range(100):  # 100 samples per crop = 2,200 rows
            n = np.random.uniform(ranges[0][0], ranges[0][1])
            p = np.random.uniform(ranges[1][0], ranges[1][1])
            k = np.random.uniform(ranges[2][0], ranges[2][1])
            temp = np.random.uniform(ranges[3][0], ranges[3][1])
            humid = np.random.uniform(ranges[4][0], ranges[4][1])
            ph = np.random.uniform(ranges[5][0], ranges[5][1])
            rain = np.random.uniform(ranges[6][0], ranges[6][1])
            
            # Add subtle Gaussian noise to make it realistic
            n = max(0, n + np.random.normal(0, 2))
            p = max(0, p + np.random.normal(0, 2))
            k = max(0, k + np.random.normal(0, 2))
            temp = max(0, temp + np.random.normal(0, 0.5))
            humid = min(100, max(0, humid + np.random.normal(0, 1)))
            ph = min(14, max(0, ph + np.random.normal(0, 0.1)))
            rain = max(0, rain + np.random.normal(0, 5))
            
            data.append([round(n), round(p), round(k), round(temp, 4), round(humid, 4), round(ph, 4), round(rain, 4), crop])
            
    df = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label'])
    df.to_csv(filepath, index=False)
    print(f"Synthetic dataset saved to {filepath} successfully with {len(df)} rows.")

def train_model():
    csv_file = 'Crop_recommendation.csv'
    if not os.path.exists(csv_file):
        generate_synthetic_data(csv_file)
        
    print("Reading dataset...")
    df = pd.read_csv(csv_file)
    
    print("Dataset Shape:", df.shape)
    print("\nDataset columns preview:\n", df.head())
    
    # Feature-Target Split
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nTraining Logistic Regression model...")
    # Increase max_iter to ensure convergence for multiclass linear logistic model
    model = LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42)
    model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Training completed. Test Set Accuracy: {accuracy * 100:.2f}%")
    
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    model_file = 'model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"Serialized model successfully saved to {model_file}")

if __name__ == '__main__':
    train_model()
