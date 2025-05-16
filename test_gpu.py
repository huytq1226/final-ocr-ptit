import torch
import easyocr

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

reader = easyocr.Reader(['vi'], gpu=True)
print("EasyOCR khởi tạo thành công với GPU!")