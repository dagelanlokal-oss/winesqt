from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memuat model dan daftar fitur yang sudah dilatih
model = joblib.load('model.joblib')
features = joblib.load('features.joblib')

# Metrik kinerja model (dari hasil train_model.py)
# Ganti dengan nilai yang Anda dapatkan dari script training
MODEL_ACCURACY = "55.35%"
MODEL_F1_SCORE = "52.67%"

# Route untuk halaman utama (form input)
@app.route('/')
def home():
    return render_template('index.html', 
                           accuracy=MODEL_ACCURACY, 
                           f1_score=MODEL_F1_SCORE,
                           features=features)

# Route untuk memproses prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Mengambil data input dari form
        input_features = [float(request.form[feature]) for feature in features]
        
        # Mengubah input menjadi DataFrame agar sesuai dengan format model
        input_df = pd.DataFrame([input_features], columns=features)
        
        # Melakukan prediksi
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        # Mengambil probabilitas tertinggi untuk prediksi
        proba = np.max(prediction_proba) * 100
        
        # Menampilkan halaman hasil
        return render_template('result.html', 
                               prediction=int(prediction[0]), 
                               confidence=f"{proba:.2f}%")

if __name__ == '__main__':
    app.run(debug=True)