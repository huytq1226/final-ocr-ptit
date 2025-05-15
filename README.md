# Dự án Nhận dạng Ký tự Quang học (OCR) với EasyOCR

Dự án này giúp chuyển đổi hình ảnh thành văn bản sử dụng EasyOCR, đặc biệt tối ưu cho tiếng Việt.

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
├── input_images/          # Thư mục chứa hình ảnh đầu vào
├── processed_images/      # Thư mục chứa hình ảnh đã xử lý
├── trained_models/        # Thư mục chứa mô hình đã huấn luyện
├── labels.txt            # File chứa nhãn cho các hình ảnh
├── main.py               # File mã nguồn chính
├── config.yaml           # File cấu hình
└── requirements.txt      # File chứa các thư viện cần thiết
```

## Sử dụng

1. Đặt hình ảnh cần xử lý vào thư mục `input_images/`

2. Chạy chương trình:

```bash
python main.py
```

3. Kết quả sẽ được lưu trong:
   - Hình ảnh đã xử lý: `processed_images/`
   - Mô hình đã huấn luyện: `trained_models/`
   - Văn bản nhận dạng được: `output.txt`

## Cấu hình

Bạn có thể điều chỉnh các tham số trong file `config.yaml`:

- Kích thước batch
- Số epoch
- Tốc độ học
- Các tham số tiền xử lý

## Xử lý lỗi thường gặp

1. Lỗi "CUDA not available":

   - Chương trình sẽ tự động chuyển sang sử dụng CPU
   - Không ảnh hưởng đến kết quả, chỉ làm chậm quá trình xử lý

2. Lỗi "Image not found":

   - Kiểm tra đường dẫn hình ảnh trong thư mục `input_images/`
   - Đảm bảo định dạng file là .jpg, .png, hoặc .jpeg

3. Lỗi "Memory error":
   - Giảm kích thước batch trong `config.yaml`
   - Xử lý ít hình ảnh hơn trong một lần chạy

## Tài liệu tham khảo

- [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo issue hoặc pull request để cải thiện dự án.

## Giấy phép

MIT License
