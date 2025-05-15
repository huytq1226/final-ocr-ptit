import os
import cv2
import yaml
import numpy as np
import easyocr
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List, Dict
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Khởi tạo bộ xử lý OCR với cấu hình từ file YAML.
        
        Args:
            config_path (str): Đường dẫn đến file cấu hình
        """
        self.config = self._load_config(config_path)
        self._create_directories()
        self.reader = self._initialize_ocr()
        
    def _load_config(self, config_path: str) -> Dict:
        """Đọc và trả về cấu hình từ file YAML."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Lỗi khi đọc file cấu hình: {e}")
            raise

    def _create_directories(self):
        """Tạo các thư mục cần thiết nếu chưa tồn tại."""
        directories = [
            self.config['paths']['input_dir'],
            self.config['paths']['processed_dir'],
            self.config['paths']['model_dir']
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _initialize_ocr(self) -> easyocr.Reader:
        """Khởi tạo EasyOCR reader với cấu hình đã cho."""
        try:
            return easyocr.Reader(
                self.config['easyocr']['languages'],
                gpu=self.config['easyocr']['gpu']
            )
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo EasyOCR: {e}")
            raise

    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Tiền xử lý hình ảnh với các bước: chuyển xám, ngưỡng, khử nhiễu.
        
        Args:
            image_path (str): Đường dẫn đến hình ảnh
            
        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]: Hình ảnh gốc và dict chứa các phiên bản đã xử lý
        """
        try:
            # Đọc hình ảnh
            original = cv2.imread(image_path)
            if original is None:
                raise ValueError(f"Không thể đọc hình ảnh: {image_path}")

            # Chuẩn hóa kích thước
            target_size = self.config['preprocessing']['target_size']
            original = cv2.resize(original, (target_size[0], target_size[1]))

            # Chuyển sang ảnh xám
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

            # Áp dụng ngưỡng
            _, thresholded = cv2.threshold(
                gray,
                self.config['preprocessing']['threshold_value'],
                255,
                cv2.THRESH_BINARY
            )

            # Khử nhiễu
            denoised = cv2.fastNlMeansDenoising(
                thresholded,
                None,
                self.config['preprocessing']['denoise_kernel_size']
            )

            processed_images = {
                'original': original,
                'gray': gray,
                'thresholded': thresholded,
                'denoised': denoised
            }

            return original, processed_images

        except Exception as e:
            logger.error(f"Lỗi khi tiền xử lý hình ảnh {image_path}: {e}")
            raise

    def save_processed_images(self, image_name: str, processed_images: Dict[str, np.ndarray]):
        """
        Lưu các phiên bản đã xử lý của hình ảnh.
        
        Args:
            image_name (str): Tên file hình ảnh
            processed_images (Dict[str, np.ndarray]): Dict chứa các phiên bản đã xử lý
        """
        try:
            base_name = Path(image_name).stem
            for process_name, image in processed_images.items():
                output_path = os.path.join(
                    self.config['paths']['processed_dir'],
                    f"{base_name}_{process_name}.jpg"
                )
                cv2.imwrite(output_path, image)
        except Exception as e:
            logger.error(f"Lỗi khi lưu hình ảnh đã xử lý: {e}")
            raise

    def visualize_preprocessing(self, image_name: str, processed_images: Dict[str, np.ndarray]):
        """
        Hiển thị các bước tiền xử lý bằng matplotlib.
        
        Args:
            image_name (str): Tên file hình ảnh
            processed_images (Dict[str, np.ndarray]): Dict chứa các phiên bản đã xử lý
        """
        try:
            plt.figure(figsize=(15, 10))
            for idx, (process_name, image) in enumerate(processed_images.items(), 1):
                plt.subplot(2, 2, idx)
                if len(image.shape) == 2:  # Ảnh xám
                    plt.imshow(image, cmap='gray')
                else:  # Ảnh màu
                    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title(f"{process_name.capitalize()}")
                plt.axis('off')

            plt.suptitle(f"Kết quả tiền xử lý cho {image_name}")
            plt.tight_layout()
            
            # Lưu hình ảnh so sánh
            output_path = os.path.join(
                self.config['paths']['processed_dir'],
                f"{Path(image_name).stem}_comparison.png"
            )
            plt.savefig(output_path)
            plt.close()

        except Exception as e:
            logger.error(f"Lỗi khi hiển thị kết quả tiền xử lý: {e}")
            raise

    def process_image(self, image_path: str) -> str:
        """
        Xử lý một hình ảnh và trả về văn bản được nhận dạng.
        
        Args:
            image_path (str): Đường dẫn đến hình ảnh
            
        Returns:
            str: Văn bản được nhận dạng
        """
        try:
            # Tiền xử lý hình ảnh
            original, processed_images = self.preprocess_image(image_path)
            
            # Lưu các phiên bản đã xử lý
            self.save_processed_images(image_path, processed_images)
            
            # Hiển thị kết quả tiền xử lý
            self.visualize_preprocessing(image_path, processed_images)
            
            # Nhận dạng văn bản từ hình ảnh đã khử nhiễu
            results = self.reader.readtext(
                processed_images['denoised'],
                detail=0,
                paragraph=True
            )
            
            # Kết hợp các kết quả thành một chuỗi
            text = ' '.join(results)
            return text

        except Exception as e:
            logger.error(f"Lỗi khi xử lý hình ảnh {image_path}: {e}")
            raise

    def process_directory(self):
        """Xử lý tất cả hình ảnh trong thư mục đầu vào."""
        try:
            input_dir = self.config['paths']['input_dir']
            output_file = self.config['paths']['output_file']
            
            # Lấy danh sách hình ảnh
            image_files = [
                f for f in os.listdir(input_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            
            if not image_files:
                logger.warning(f"Không tìm thấy hình ảnh trong {input_dir}")
                return
            
            # Xử lý từng hình ảnh
            results = []
            for image_file in tqdm(image_files, desc="Đang xử lý hình ảnh"):
                image_path = os.path.join(input_dir, image_file)
                try:
                    text = self.process_image(image_path)
                    results.append(f"{image_file}\t{text}")
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý {image_file}: {e}")
                    continue
            
            # Lưu kết quả
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(results))
            
            logger.info(f"Đã xử lý xong {len(results)} hình ảnh")
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý thư mục: {e}")
            raise

def main():
    """Hàm chính của chương trình."""
    try:
        # Khởi tạo bộ xử lý OCR
        processor = OCRProcessor()
        
        # Xử lý thư mục hình ảnh
        processor.process_directory()
        
        logger.info("Hoàn thành xử lý!")
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình xử lý: {e}")
        raise

if __name__ == "__main__":
    main() 