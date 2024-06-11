import joblib
import sklearn

# Đường dẫn tới tệp mô hình
model_path = 'D:\DoAn4\project\knn_model.pkl'

# Đọc mô hình từ tệp
model = joblib.load(model_path)

# In phiên bản của scikit-learn
print("Phiên bản scikit-learn:", sklearn.__version__)