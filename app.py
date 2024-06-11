from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import joblib
import numpy as np
import librosa
import soundfile as sf
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_features(file_path):
    try:
        # Load tệp âm thanh và chia thành cửa sổ tín hiệu
        y, sr = librosa.load(file_path, sr=22050)
        n_fft = 512 # Kích thước cửa sổ FFT
        hop_length = 256  # Khoảng cách giữa các frame
        
        # Tính STFT và các đặc trưng liên quan đến âm sắc
        # Đặc trưng 1 : Spectral Centroid
        stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        spectral_centroid = librosa.feature.spectral_centroid(S=stft**2, sr=sr)[0]
        # Đặc trưng 2: Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        # Đặc trưng 3: Spectral Flux
        flux = librosa.onset.onset_strength(y=y, sr=sr)
        # Đặc trưng 4: Zero-crossings
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        # Đặc trưng 5: Low-Energy
        rms = librosa.feature.rms(y=y)[0]
        # Đặc trưng 6: Các hệ số MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length, n_mels=128)
        
        # Trích xuất các giá trị trung bình và độ lệch chuẩn của các đặc trưng
        mean_spectral_centroid = np.mean(spectral_centroid)
        std_spectral_centroid = np.std(spectral_centroid)
        mean_rolloff = np.mean(rolloff)
        std_rolloff = np.std(rolloff)
        mean_flux = np.mean(flux)
        std_flux = np.std(flux)
        mean_zero_crossing_rate = np.mean(zero_crossing_rate)
        std_zero_crossing_rate = np.std(zero_crossing_rate)
        mean_low_energy = np.mean(rms)
        low_energy = np.mean(rms < mean_low_energy)
        mfcc_mean = np.mean(mfcc, axis=1)[:5]  # Chỉ lấy 5 đặc trưng đầu của MFCC
        mfcc_std = np.std(mfcc, axis=1)[:5]    # Chỉ lấy 5 đặc trưng đầu của MFCC
        
        # Tạo vector đặc trưng bao gồm cả MFCC
        feature_vector = np.array([
            mean_spectral_centroid, std_spectral_centroid,
            mean_rolloff, std_rolloff,
            mean_flux, std_flux,
            mean_zero_crossing_rate, std_zero_crossing_rate,
            mean_low_energy, low_energy
        ])
        
        # Thêm MFCC vào vector đặc trưng
        feature_vector = np.concatenate((feature_vector, mfcc_mean, mfcc_std))
        
        return feature_vector.reshape(1, -1)
    except Exception as e:
        print("Error extracting features:", str(e))
        return None


    
def load_model_and_scaler():
    scaler_path = 'feature_scaler.pkl'
    model_path = 'knn_model.pkl'
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return scaler, model
scaler, model = load_model_and_scaler()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_audio():
    if 'audioFile' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    audio_file = request.files['audioFile']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        audio_file.save(file_path)

        print("File saved at:", file_path)  # In đường dẫn của tệp âm thanh đã lưu

        features = extract_features(file_path)
        print("Extracted features:", features)  # In các đặc trưng được trích xuất

        # Kiểm tra kích thước và nội dung của features trước khi chuẩn hóa
        if features is None or features.size == 0:
            raise ValueError("No features extracted from audio file")
        
        print("Feature shape:", features.shape)  # In kích thước của các đặc trưng
        features_scaled = scaler.transform(features)
        print("Scaled features:", features_scaled)  # In các đặc trưng đã được chuẩn hóa

        predicted_label = model.predict(features_scaled)
        print("Predicted label:", predicted_label)  # In nhãn dự đoán

        predicted_prob = model.predict_proba(features_scaled) # dòng này chỉ dùng cho KNN
        print("Predicted probabilities:", predicted_prob)  # In xác suất dự đoán

        return jsonify({
            'genre': predicted_label[0], 
            'probability': np.max(predicted_prob) * 100,
            'file_path': f'/uploads/{audio_file.filename}'
        })  
    except Exception as e:
        print("Error during classification:", str(e))  # In thông báo lỗi
        return jsonify({'error': 'Internal Server Error: ' + str(e)}), 500




@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
