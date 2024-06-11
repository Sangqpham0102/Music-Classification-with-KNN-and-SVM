# Phân loại Âm nhạc Truyền thống Việt Nam

- Dự án cá nhân này tập trung vào việc phân loại các thể loại âm nhạc truyền thống Việt Nam
- Dựa trên các đặc trưng âm sắc như Spectral Centroid, Rolloff, Flux, Zero Crossing, Low Energy và MFCC.
- Hoàn thành 10/06/2024.

## Mục lục
- [Giới thiệu](#giới-thiệu)
- [Cài đặt](#cài-đặt)
- [Các phiên bản thư viện sử dụng](#các-phiên-bản-thư-viện-sử-dụng)
- [Quá trình thực hiện](#quá-trình-thực-hiện)
- [Tham khảo](#tham-khảo)

## Giới thiệu
- Dự án này nhằm xây dựng một mô hình máy học để phân loại các thể loại âm nhạc truyền thống Việt Nam bao gồm: Cailuong, Catru, Chauvan, Cheo, và Xam.
- Chúng tôi sử dụng các thư viện Python để xử lý âm thanh và học máy như librosa để trích xuất đặc trưng và TensorFlow/Keras để xây dựng và huấn luyện mô hình.

## Cài đặt
Để thiết lập dự án này, bạn cần cài đặt Python 3.6+ và các thư viện cần thiết sau:

1. Clone repository:
   ```bash
   https://github.com/Sangqpham0102/Machine-learning-project.git
   cd project
2. Cài đặc thư viện
pip install -r requirements.txt
## Quá trình thực hiện
- Chuẩn bị dữ liệu
- Trích xuất đặc trưng
- Huấn luyện mô hình
- Đánh giá
- Dự đoán thể loại
Chi tiết được hiện trong file ,Audi_Training.ipynb
## Tham khảo
[1] LTPhat/ Phân loại Việt-Truyền thống-Âm nhạc-Phân loại, https://github.com/LTPhat/Vietnamese-Traditional-Music-Classification

[2] Thư viện Librosa, https://librosa.org/doc/latest/index.html

[3] TensorFlow, https://www.tensorflow.org/
