from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from werkzeug.utils import secure_filename
import base64
from PIL import Image
import io
import urllib.request
from urllib.parse import urlparse

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CSV_FOLDER'] = 'csv_data'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Membuat folder jika belum ada
for folder in [app.config['UPLOAD_FOLDER'], app.config['CSV_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Ekstensi file yang diizinkan
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_CSV_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_csv_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_CSV_EXTENSIONS

def download_image(url, save_path):
    """Download gambar dari URL"""
    try:
        urllib.request.urlretrieve(url, save_path)
        return True
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return False

def extract_features(image_path):
    """
    Ekstrak fitur dari gambar untuk model Random Forest
    Menggunakan histogram warna dan tekstur sederhana
    """
    try:
        # Baca gambar
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Resize gambar ke ukuran standar
        img = cv2.resize(img, (64, 64))
        
        # Konversi ke HSV untuk analisis warna yang lebih baik
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Ekstrak histogram warna
        hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [60], [0, 256])
        
        # Gabungkan histogram
        features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
        
        # Tambahkan fitur statistik sederhana
        mean_rgb = np.mean(img.reshape(-1, 3), axis=0)
        std_rgb = np.std(img.reshape(-1, 3), axis=0)
        
        # Gabungkan semua fitur
        all_features = np.concatenate([features, mean_rgb, std_rgb])
        
        return all_features
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def process_csv_images(csv_path, image_column='image_path', label_column='label'):
    """
    Memproses semua gambar dari CSV untuk training model
    """
    try:
        # Baca CSV
        df = pd.read_csv(csv_path)
        
        features_list = []
        labels_list = []
        processed_count = 0
        
        print(f"Processing {len(df)} images from CSV...")
        
        for idx, row in df.iterrows():
            try:
                image_path = row[image_column]
                label = row[label_column] if label_column in df.columns else 0
                
                # Jika path adalah URL, download dulu
                if image_path.startswith('http'):
                    local_path = os.path.join(app.config['CSV_FOLDER'], f"temp_img_{idx}.jpg")
                    if download_image(image_path, local_path):
                        features = extract_features(local_path)
                        os.remove(local_path)  # Hapus file temporary
                    else:
                        continue
                else:
                    # Path lokal
                    if not os.path.exists(image_path):
                        print(f"Image not found: {image_path}")
                        continue
                    features = extract_features(image_path)
                
                if features is not None:
                    features_list.append(features)
                    labels_list.append(label)
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        print(f"Processed {processed_count} images...")
                        
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        print(f"Successfully processed {processed_count} images")
        return np.array(features_list), np.array(labels_list)
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None

def train_model_from_csv(csv_path, image_column='image_path', label_column='label'):
    """
    Latih model Random Forest dari data CSV
    """
    X, y = process_csv_images(csv_path, image_column, label_column)
    
    if X is None or len(X) == 0:
        print("No valid data found for training")
        return None
        
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Latih model Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Evaluasi model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Simpan model
    joblib.dump(rf_model, 'rf_model.pkl')
    
    # Simpan label mapping
    unique_labels = sorted(list(set(y)))
    label_mapping = {i: f"Class_{label}" for i, label in enumerate(unique_labels)}
    joblib.dump(label_mapping, 'label_mapping.pkl')
    
    return rf_model, label_mapping

def create_sample_model():
    """
    Membuat model Random Forest sample dengan data dummy
    """
    np.random.seed(42)
    n_samples = 1000
    n_features = 176  # Sesuai dengan jumlah fitur yang diekstrak
    
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 5, n_samples)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Sample model accuracy: {accuracy:.2f}")
    
    joblib.dump(rf_model, 'rf_model.pkl')
    
    return rf_model

# Load atau buat model
try:
    model = joblib.load('rf_model.pkl')
    try:
        CLASS_LABELS = joblib.load('label_mapping.pkl')
    except:
        CLASS_LABELS = {
            0: "Kategori A", 1: "Kategori B", 2: "Kategori C",
            3: "Kategori D", 4: "Kategori E"
        }
    print("Model loaded successfully")
except:
    print("Creating new sample model...")
    model = create_sample_model()
    CLASS_LABELS = {
        0: "Kategori A", 1: "Kategori B", 2: "Kategori C",
        3: "Kategori D", 4: "Kategori E"
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_csv')
def upload_csv_page():
    return render_template('upload_csv.html')

@app.route('/train_from_csv', methods=['POST'])
def train_from_csv():
    if 'csv_file' not in request.files:
        flash('Tidak ada file CSV yang dipilih')
        return redirect(url_for('upload_csv_page'))
    
    file = request.files['csv_file']
    
    if file.filename == '' or not allowed_csv_file(file.filename):
        flash('File CSV tidak valid')
        return redirect(url_for('upload_csv_page'))
    
    try:
        filename = secure_filename(file.filename)
        csv_path = os.path.join(app.config['CSV_FOLDER'], filename)
        file.save(csv_path)
        
        # Dapatkan nama kolom dari form
        image_column = request.form.get('image_column', 'image_path')
        label_column = request.form.get('label_column', 'label')
        
        # Train model dari CSV
        global model, CLASS_LABELS
        result = train_model_from_csv(csv_path, image_column, label_column)
        
        if result is not None:
            model, CLASS_LABELS = result
            flash(f'Model berhasil dilatih dari {filename}!')
        else:
            flash('Gagal melatih model dari CSV')
        
        # Hapus file CSV setelah diproses
        os.remove(csv_path)
        
    except Exception as e:
        flash(f'Error memproses CSV: {str(e)}')
    
    return redirect(url_for('upload_csv_page'))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('Tidak ada file yang dipilih')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('Tidak ada file yang dipilih')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Ekstrak fitur dari gambar
            features = extract_features(filepath)
            
            if features is None:
                flash('Error: Tidak dapat memproses gambar')
                return redirect(url_for('index'))
            
            # Prediksi menggunakan model Random Forest
            prediction = model.predict([features])[0]
            probability = model.predict_proba([features])[0]
            confidence = max(probability) * 100
            
            # Konversi gambar ke base64 untuk ditampilkan
            with open(filepath, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Hapus file setelah diproses
            os.remove(filepath)
            
            result = {
                'prediction': CLASS_LABELS.get(prediction, f"Kelas {prediction}"),
                'confidence': round(confidence, 2),
                'image': img_base64,
                'filename': filename
            }
            
            return render_template('result.html', result=result)
            
        except Exception as e:
            flash(f'Error memproses file: {str(e)}')
            return redirect(url_for('index'))
    
    else:
        flash('Tipe file tidak diizinkan. Gunakan PNG, JPG, JPEG, GIF, atau BMP')
        return redirect(url_for('index'))

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Prediksi batch dari multiple files"""
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        flash('Tidak ada file yang dipilih')
        return redirect(url_for('index'))
    
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                features = extract_features(filepath)
                
                if features is not None:
                    prediction = model.predict([features])[0]
                    probability = model.predict_proba([features])[0]
                    confidence = max(probability) * 100
                    
                    with open(filepath, "rb") as img_file:
                        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    results.append({
                        'filename': filename,
                        'prediction': CLASS_LABELS.get(prediction, f"Kelas {prediction}"),
                        'confidence': round(confidence, 2),
                        'image': img_base64
                    })
                
                os.remove(filepath)
                
            except Exception as e:
                print(f"Error processing {file.filename}: {e}")
                continue
    
    return render_template('batch_results.html', results=results)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint untuk prediksi"""
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang ditemukan'}), 400
    
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'File tidak valid'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        features = extract_features(filepath)
        
        if features is None:
            return jsonify({'error': 'Tidak dapat memproses gambar'}), 400
        
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0]
        confidence = max(probability) * 100
        
        os.remove(filepath)
        
        return jsonify({
            'prediction': CLASS_LABELS.get(prediction, f"Kelas {prediction}"),
            'confidence': round(confidence, 2),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    """Informasi tentang model yang sedang digunakan"""
    try:
        model_info = {
            'n_estimators': model.n_estimators,
            'n_features': model.n_features_in_,
            'n_classes': len(CLASS_LABELS),
            'classes': CLASS_LABELS
        }
        return render_template('model_info.html', info=model_info)
    except:
        return render_template('model_info.html', info=None)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Gunakan port dari Railway
    app.run(debug=False, host='0.0.0.0', port=port)
