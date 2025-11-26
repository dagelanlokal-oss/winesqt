import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
import joblib

# 1. Load Data
try:
    df = pd.read_csv('wineQT.csv')
except FileNotFoundError:
    print("Error: Pastikan file wineQT.csv ada di folder yang sama dengan script ini.")
    exit()

# 2. Tentukan Fitur (X) dan Target (y)
# Kolom 'Id' tidak digunakan, dan 'quality' adalah target kita
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol']
target = 'quality'

X = df[features]
y = df[target]

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Latih Model Gaussian Naive Bayes
print("Melatih model Gaussian Naive Bayes...")
model = GaussianNB()
model.fit(X_train, y_train)

# 5. Evaluasi Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted') # 'weighted' baik untuk kelas tidak seimbang

print(f"\nModel berhasil dilatih!")
print(f"Akurasi Model: {accuracy:.4f}")
print(f"F1-Score Model: {f1:.4f}")

# 6. Simpan Model dan Fitur
# Menyimpan model ke dalam file
joblib.dump(model, 'model.joblib')
# Menyimpan nama fitur untuk memastikan input user sesuai urutan
joblib.dump(features, 'features.joblib')

print("\nModel dan daftar fitur telah disimpan ke 'model.joblib' dan 'features.joblib'")