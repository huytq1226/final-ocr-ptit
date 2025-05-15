# Dự án Nhận dạng Ký tự Quang học (OCR) với EasyOCR

Dự án này giúp chuyển đổi hình ảnh thành văn bản sử dụng EasyOCR, đặc biệt tối ưu cho tiếng Việt. Dự án bao gồm hai phần chính: huấn luyện mô hình và triển khai web interface.

## Yêu cầu hệ thống

- Python 3.8 trở lên
- CPU (không yêu cầu GPU)
- 4GB RAM trở lên

## Cài đặt

1. Clone repository này về máy:

```bash
git clone <repository-url>
cd ocr_project
```

2. Tạo môi trường ảo Python (khuyến nghị):

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## Cấu trúc thư mục

```
ocr_project/
├── ocr_data/              # Thư mục chứa dữ liệu huấn luyện
│   ├── input_images/     # Hình ảnh đầu vào để huấn luyện
│   ├── processed_images/ # Hình ảnh đã được tiền xử lý
│   └── labels.txt       # File chứa nhãn cho các hình ảnh
├── trained_models/       # Thư mục chứa các mô hình đã huấn luyện
│   └── model_YYYYMMDD_HHMMSS/  # Thư mục mô hình với timestamp
├── static/              # Thư mục chứa file tĩnh cho web app
│   └── uploads/        # Thư mục lưu hình ảnh tải lên qua web
├── templates/          # Thư mục chứa template HTML
│   └── index.html     # Giao diện web chính
├── main.py            # Script huấn luyện mô hình
├── deploy_model.py    # Script triển khai web interface
├── config.yaml        # File cấu hình
└── requirements.txt   # File chứa các thư viện cần thiết
```

## Sử dụng

### 1. Huấn luyện mô hình

1. Đặt hình ảnh cần huấn luyện vào thư mục `ocr_data/input_images/`
2. Tạo file `ocr_data/labels.txt` với định dạng:
   ```
   image_name.jpg    text_content
   image_name2.jpg   text_content2
   ```
3. Chạy script huấn luyện:
   ```bash
   python main.py
   ```
4. Mô hình sẽ được lưu trong thư mục `trained_models/` với timestamp

### 2. Triển khai Web Interface

1. Chạy ứng dụng web:
   ```bash
   python deploy_model.py
   ```
2. Truy cập http://localhost:5000 trong trình duyệt
3. Tải lên hình ảnh và nhận kết quả nhận dạng

## Cấu hình

Bạn có thể điều chỉnh các tham số trong file `config.yaml`:

- Cấu hình EasyOCR:
  - Ngôn ngữ hỗ trợ
  - Sử dụng GPU
- Cấu hình tiền xử lý:
  - Kích thước ảnh đích
  - Ngưỡng xử lý
  - Tham số khử nhiễu
- Cấu hình huấn luyện:
  - Tỷ lệ chia tập train/test
  - Số epoch
  - Các tham số khác

## Xử lý lỗi thường gặp

1. Lỗi "CUDA not available":

   - Chương trình sẽ tự động chuyển sang sử dụng CPU
   - Không ảnh hưởng đến kết quả, chỉ làm chậm quá trình xử lý

2. Lỗi "Image not found":

   - Kiểm tra đường dẫn hình ảnh trong thư mục `ocr_data/input_images/`
   - Đảm bảo định dạng file là .jpg, .png, hoặc .jpeg

3. Lỗi "Memory error":

   - Giảm kích thước ảnh trong `config.yaml`
   - Xử lý ít hình ảnh hơn trong một lần chạy

4. Lỗi "Model not found":
   - Đảm bảo đã chạy `main.py` để tạo mô hình
   - Kiểm tra thư mục `trained_models/` có chứa mô hình

## Tính năng

- Hỗ trợ nhận dạng tiếng Việt
- Giao diện web thân thiện với người dùng
- Hỗ trợ kéo thả file
- Hiển thị kết quả trực quan
- Lưu lịch sử huấn luyện và đồ thị
- Tiền xử lý hình ảnh tự động

## Tài liệu tham khảo

- [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Bootstrap Documentation](https://getbootstrap.com/)

## Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo issue hoặc pull request để cải thiện dự án.

## Giấy phép

MIT License
