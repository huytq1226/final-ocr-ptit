from flask import Flask, render_template, request, jsonify
import os
from pathlib import Path
import cv2
import numpy as np
import easyocr
import yaml
from datetime import datetime
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Cấu hình
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Tạo thư mục uploads nếu chưa tồn tại
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_ocr_model():
    """Tải mô hình OCR đã huấn luyện."""
    try:
        # Tìm mô hình mới nhất trong thư mục trained_models
        model_dir = Path('trained_models')
        if not model_dir.exists():
            return None
            
        # Lấy thư mục mô hình mới nhất
        model_paths = sorted(model_dir.glob('model_*'), key=lambda x: x.stat().st_mtime, reverse=True)
        if not model_paths:
            return None
            
        latest_model = model_paths[0]
        config_path = latest_model / 'config.yaml'
        
        # Đọc cấu hình
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # Khởi tạo EasyOCR reader
        reader = easyocr.Reader(
            config['easyocr']['languages'],
            gpu=config['easyocr']['gpu']
        )
        
        return reader
        
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình: {e}")
        return None

# Tải mô hình khi khởi động ứng dụng
ocr_reader = load_ocr_model()

@app.route('/')
def index():
    """Trang chủ."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Xử lý tải lên hình ảnh và thực hiện OCR."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Không tìm thấy file'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Không có file được chọn'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Định dạng file không được hỗ trợ'}), 400
            
        # Lưu file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Đọc hình ảnh
        image = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Không thể đọc hình ảnh'}), 400
            
        # Thực hiện OCR
        if ocr_reader is None:
            return jsonify({'error': 'Mô hình OCR chưa được tải'}), 500
            
        results = ocr_reader.readtext(image, detail=0, paragraph=True)
        text = ' '.join(results)
        
        return jsonify({
            'success': True,
            'text': text,
            'image_url': f'/static/uploads/{filename}'
        })
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý file: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)