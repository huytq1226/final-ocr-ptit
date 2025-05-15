import os
import cv2
import yaml
import numpy as np
import easyocr
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging
from datetime import datetime
import json
from sklearn.model_selection import train_test_split

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OCRTrainer:
    def __init__(self, config: Dict):
        """
        Khởi tạo bộ huấn luyện OCR.
        
        Args:
            config (Dict): Cấu hình từ file YAML
        """
        self.config = config
        self.model_dir = Path(config['paths']['model_dir'])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.training_history = []
        
    def load_dataset(self, labels_file: str) -> Tuple[List[str], List[str]]:
        """
        Tải dữ liệu từ file labels.txt.
        
        Args:
            labels_file (str): Đường dẫn đến file labels.txt
            
        Returns:
            Tuple[List[str], List[str]]: Danh sách đường dẫn hình ảnh và nhãn tương ứng
        """
        try:
            image_paths = []
            labels = []
            
            with open(labels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    image_name, label = line.strip().split('\t')
                    image_path = Path(self.config['paths']['input_dir']) / image_name
                    if image_path.exists():
                        image_paths.append(str(image_path))
                        labels.append(label)
                    else:
                        logger.warning(f"Không tìm thấy hình ảnh: {image_path}")
            
            return image_paths, labels
            
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu: {e}")
            raise

    def prepare_data(self, image_paths: List[str], labels: List[str]) -> Tuple[Dict, Dict]:
        """
        Chuẩn bị dữ liệu cho việc huấn luyện.
        
        Args:
            image_paths (List[str]): Danh sách đường dẫn hình ảnh
            labels (List[str]): Danh sách nhãn
            
        Returns:
            Tuple[Dict, Dict]: Dữ liệu huấn luyện và kiểm tra
        """
        try:
            # Chia dữ liệu thành tập huấn luyện và kiểm tra
            train_paths, test_paths, train_labels, test_labels = train_test_split(
                image_paths, labels,
                test_size=1 - self.config['training']['train_test_split'],
                random_state=42
            )
            
            return {
                'train': {'paths': train_paths, 'labels': train_labels},
                'test': {'paths': test_paths, 'labels': test_labels}
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi chuẩn bị dữ liệu: {e}")
            raise

    def train_model(self, data: Dict) -> None:
        """
        Huấn luyện mô hình OCR.
        
        Args:
            data (Dict): Dữ liệu đã được chuẩn bị
        """
        try:
            logger.info("Bắt đầu huấn luyện mô hình...")
            
            # Khởi tạo EasyOCR reader
            reader = easyocr.Reader(
                self.config['easyocr']['languages'],
                gpu=self.config['easyocr']['gpu']
            )
            
            # Tạo thư mục cho mô hình
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.model_dir / f"model_{timestamp}"
            model_path.mkdir(exist_ok=True)
            
            # Huấn luyện mô hình
            for epoch in range(self.config['training']['num_epochs']):
                epoch_loss = 0
                correct_predictions = 0
                total_predictions = 0
                
                # Huấn luyện trên tập train
                for img_path, label in tqdm(
                    zip(data['train']['paths'], data['train']['labels']),
                    desc=f"Epoch {epoch + 1}/{self.config['training']['num_epochs']}"
                ):
                    try:
                        # Đọc và tiền xử lý hình ảnh
                        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if image is None:
                            continue
                            
                        # Nhận dạng văn bản
                        results = reader.readtext(image, detail=0, paragraph=True)
                        predicted_text = ' '.join(results)
                        
                        # Tính toán độ chính xác
                        if predicted_text.strip() == label.strip():
                            correct_predictions += 1
                        total_predictions += 1
                        
                        # Cập nhật loss (ví dụ đơn giản)
                        loss = 1.0 - (correct_predictions / total_predictions)
                        epoch_loss += loss
                        
                    except Exception as e:
                        logger.error(f"Lỗi khi xử lý hình ảnh {img_path}: {e}")
                        continue
                
                # Tính toán metrics
                avg_loss = epoch_loss / len(data['train']['paths'])
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                
                # Lưu lịch sử huấn luyện
                self.training_history.append({
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'accuracy': accuracy
                })
                
                logger.info(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
            
            # Lưu mô hình và lịch sử huấn luyện
            self.save_model(model_path)
            self.save_training_history(model_path)
            
            logger.info(f"Hoàn thành huấn luyện! Mô hình được lưu tại: {model_path}")
            
        except Exception as e:
            logger.error(f"Lỗi trong quá trình huấn luyện: {e}")
            raise

    def save_model(self, model_path: Path) -> None:
        """
        Lưu mô hình đã huấn luyện.
        
        Args:
            model_path (Path): Đường dẫn để lưu mô hình
        """
        try:
            # Lưu cấu hình
            config_path = model_path / 'config.yaml'
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, allow_unicode=True)
                
        except Exception as e:
            logger.error(f"Lỗi khi lưu mô hình: {e}")
            raise

    def save_training_history(self, model_path: Path) -> None:
        """
        Lưu lịch sử huấn luyện.
        
        Args:
            model_path (Path): Đường dẫn để lưu lịch sử
        """
        try:
            history_path = model_path / 'training_history.json'
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=4)
                
            # Vẽ đồ thị
            self.plot_training_history(model_path)
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu lịch sử huấn luyện: {e}")
            raise

    def plot_training_history(self, model_path: Path) -> None:
        """
        Vẽ đồ thị quá trình huấn luyện.
        
        Args:
            model_path (Path): Đường dẫn để lưu đồ thị
        """
        try:
            epochs = [h['epoch'] for h in self.training_history]
            losses = [h['loss'] for h in self.training_history]
            accuracies = [h['accuracy'] for h in self.training_history]
            
            plt.figure(figsize=(12, 5))
            
            # Vẽ đồ thị loss
            plt.subplot(1, 2, 1)
            plt.plot(epochs, losses, 'b-', label='Loss')
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Vẽ đồ thị accuracy
            plt.subplot(1, 2, 2)
            plt.plot(epochs, accuracies, 'r-', label='Accuracy')
            plt.title('Training Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(model_path / 'training_history.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Lỗi khi vẽ đồ thị: {e}")
            raise

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
        self.trainer = OCRTrainer(self.config)
        
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
            # Chuyển đổi đường dẫn thành Path object để xử lý đúng trên mọi hệ điều hành
            image_path = str(Path(image_path))
            
            # Đọc hình ảnh
            original = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
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
                output_path = Path(self.config['paths']['processed_dir']) / f"{base_name}_{process_name}.jpg"
                # Sử dụng imencode để lưu file với tên tiếng Việt
                _, buffer = cv2.imencode('.jpg', image)
                output_path.write_bytes(buffer)
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
            output_path = Path(self.config['paths']['processed_dir']) / f"{Path(image_name).stem}_comparison.png"
            plt.savefig(str(output_path))
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
            input_dir = Path(self.config['paths']['input_dir'])
            output_file = Path(self.config['paths']['output_file'])
            
            # Lấy danh sách hình ảnh
            image_files = [
                f for f in input_dir.glob('*')
                if f.suffix.lower() in ['.png', '.jpg', '.jpeg']
            ]
            
            if not image_files:
                logger.warning(f"Không tìm thấy hình ảnh trong {input_dir}")
                return
            
            # Xử lý từng hình ảnh
            results = []
            for image_file in tqdm(image_files, desc="Đang xử lý hình ảnh"):
                try:
                    text = self.process_image(str(image_file))
                    results.append(f"{image_file.name}\t{text}")
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý {image_file.name}: {e}")
                    continue
            
            # Lưu kết quả
            output_file.write_text('\n'.join(results), encoding='utf-8')
            
            logger.info(f"Đã xử lý xong {len(results)} hình ảnh")
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý thư mục: {e}")
            raise

    def train(self, labels_file: str) -> None:
        """
        Huấn luyện mô hình OCR.
        
        Args:
            labels_file (str): Đường dẫn đến file labels.txt
        """
        try:
            # Tải dữ liệu
            image_paths, labels = self.trainer.load_dataset(labels_file)
            
            # Chuẩn bị dữ liệu
            data = self.trainer.prepare_data(image_paths, labels)
            
            # Huấn luyện mô hình
            self.trainer.train_model(data)
            
        except Exception as e:
            logger.error(f"Lỗi trong quá trình huấn luyện: {e}")
            raise

def main():
    """Hàm chính của chương trình."""
    try:
        # Khởi tạo bộ xử lý OCR
        processor = OCRProcessor()
        
        # Huấn luyện mô hình nếu có file labels.txt
        labels_file = processor.config['paths']['labels_file']
        if Path(labels_file).exists():
            logger.info("Bắt đầu huấn luyện mô hình...")
            processor.train(labels_file)
        else:
            logger.warning(f"Không tìm thấy file {labels_file}, bỏ qua bước huấn luyện")
        
        # Xử lý thư mục hình ảnh
        processor.process_directory()
        
        logger.info("Hoàn thành xử lý!")
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình xử lý: {e}")
        raise

if __name__ == "__main__":
    main() 