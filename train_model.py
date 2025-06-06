import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import shutil

# --- Fungsi untuk membaca gambar dan label ---
def load_images_from_folder(folder_path, img_size=(100, 100)):
    images = []
    labels = []
    for label_name in os.listdir(folder_path):
        label_folder = os.path.join(folder_path, label_name)
        if not os.path.isdir(label_folder):
            continue
        for img_file in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = img.flatten()  # konversi jadi vektor 1 dimensi
                images.append(img)
                labels.append(label_name)
    return np.array(images), np.array(labels)

# --- Fungsi untuk menampilkan distribusi kelas ---
def print_class_distribution(labels, title="Distribusi Kelas"):
    label_counts = Counter(labels)
    total = sum(label_counts.values())
    print(f"\n📊 {title}")
    for label, count in label_counts.items():
        percent = (count / total) * 100
        print(f"{label}: {count} data ({percent:.2f}%)")

# --- Load data training, validasi, dan testing ---
X_train, y_train = load_images_from_folder("Train")
X_val, y_val = load_images_from_folder("Validation")
X_test, y_test = load_images_from_folder("Test")

# --- Tampilkan distribusi kelas ---
print_class_distribution(y_train, "Data Train")
print_class_distribution(y_val, "Data Validasi")
print_class_distribution(y_test, "Data Test")

# --- Encode label ke angka ---
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

# --- Buat dan latih model Random Forest ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train_enc)

# --- Evaluasi pada data validasi ---
print("\n📈 Evaluasi - Validasi:")
y_val_pred = model.predict(X_val)
print("Akurasi:", accuracy_score(y_val_enc, y_val_pred))
print(classification_report(y_val_enc, y_val_pred, target_names=le.classes_))

# --- Evaluasi pada data testing ---
print("\n📈 Evaluasi - Testing:")
y_test_pred = model.predict(X_test)
print("Akurasi:", accuracy_score(y_test_enc, y_test_pred))
print(classification_report(y_test_enc, y_test_pred, target_names=le.classes_))

# --- Simpan model dan label encoder ---
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model_rf.pkl")
joblib.dump(le, "model/label_encoder.pkl")
print("✅ Model dan encoder disimpan di folder 'model/'")

# --- Visualisasi Confusion Matrix ---
cm = confusion_matrix(y_test_enc, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Test Data")
plt.tight_layout()
plt.show()

# --- Visualisasi Feature Importances (Top 10) ---
importances = model.feature_importances_
indices = np.argsort(importances)[-10:][::-1]  # Top 10 pixel

plt.figure(figsize=(8, 5))
plt.title("Top 10 Feature Importances (Pixel)")
bars = plt.barh(range(len(indices)), importances[indices][::-1],
                color=plt.cm.viridis(np.linspace(0, 1, len(indices))))
plt.yticks(range(len(indices)), [f'Pixel {i}' for i in indices[::-1]])
plt.xlabel("Tingkat Kepentingan")
plt.tight_layout()
plt.show()

# --- Buat salinan model dan label encoder untuk digunakan di Flask ---
shutil.copy("model/model_rf.pkl", "model_random_forest.pkl")
shutil.copy("model/label_encoder.pkl", "label_encoder.pkl")
print("💾 Model dan label encoder siap digunakan untuk aplikasi Flask.")