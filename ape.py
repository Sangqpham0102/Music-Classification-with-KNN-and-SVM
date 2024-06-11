import numpy as np
import os
import librosa
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Load mô hình KNN và bộ chuẩn hóa từ các tệp pickle đã lưu trước đó
load_model = joblib.load('D:\\DoAn4\\project\\cd\\knn_model.pkl')
scaler = joblib.load('D:\\DoAn4\\project\cd\\feature_scaler.pkl')

def extract_features(file_path):
    try:
        # Load tệp âm thanh và chia thành cửa sổ tín hiệu
        y, sr = librosa.load(file_path, sr=22050)
        n_fft = 2048  # Kích thước cửa sổ FFT
        hop_length = 512  # Khoảng cách giữa các frame
        
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
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length, n_mels=96)
        
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
            mean_low_energy,low_energy,
        ])
        
        # Thêm MFCC vào vector đặc trưng
        feature_vector = np.concatenate((feature_vector, mfcc_mean, mfcc_std))
        
        return feature_vector
    # Xử lý ngoại lệ
    except Exception as e:
        print("Error extracting features:", str(e))
        return None
# Danh sách các đường dẫn đến các file âm thanh bạn muốn dự đoán
file_paths = [
    "D:\\DoAn4\\project\\uploads\\XamHueTinh-HaThiCau-HATXAM.mp3",
    "D:\\DoAn4\\project\\uploads\\LenhTruyNa-VuongLinh-CAILUONG.mp3",
    "D:\\DoAn4\\project\\uploads\\ThuocPhien-HaThiCau-HATXAM.mp3",
    "D:\\DoAn4\\project\\uploads\\TuongTienTuu-QuachThiHo-CATRU.wav",
    "D:\\DoAn4\\project\\uploads\\Chauvan.393.wav"

   
    # Thêm các đường dẫn khác nếu cần
]

audio_formats = ['.wav', '.mp3', '.ogg']  # Thêm các định dạng khác nếu cần

# Duyệt qua từng đường dẫn và dự đoán thể loại của từng file
for file_path in file_paths:
    # Kiểm tra định dạng của tệp âm thanh
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() not in audio_formats:
        print(f"File {file_path} không phải là định dạng âm thanh được hỗ trợ.")
        continue

    # Trích xuất đặc trưng của file âm thanh
    song_features = extract_features(file_path)
    # Chuẩn hóa đặc trưng của file âm thanh
    song_features_scaled = scaler.transform(song_features.reshape(1, -1))
    # Dự đoán thể loại của file âm thanh bằng mô hình đã được tải
    predicted_prob = load_model.predict_proba(song_features_scaled)
    predicted_label = load_model.predict(song_features_scaled)
    # Hiển thị kết quả dự đoán
    print("Thể loại nhạc:", predicted_label[0])
    print("Độ chính xác:", np.max(predicted_prob) * 100, "%") # knn
