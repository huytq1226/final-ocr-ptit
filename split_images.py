import os
import shutil

# Đường dẫn đến thư mục chứa ảnh
input_dir = "ocr_data/input_images/"

# Số ảnh mỗi thư mục con
images_per_folder = 100

# Đảm bảo thư mục input_images tồn tại
if not os.path.exists(input_dir):
    print(f"Thư mục {input_dir} không tồn tại. Vui lòng kiểm tra lại!")
    exit()

# Lấy danh sách tất cả các tệp ảnh
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]
total_images = len(image_files)

if total_images == 0:
    print("Không tìm thấy ảnh trong thư mục input_images!")
    exit()

print(f"Tổng số ảnh: {total_images}")

# Tính số thư mục con cần tạo
num_subfolders = (total_images + images_per_folder - 1) // images_per_folder

# Tạo và di chuyển ảnh vào các thư mục con
for i in range(num_subfolders):
    # Tạo thư mục con (subfolder_0, subfolder_1, ...)
    subfolder_name = f"subfolder_{i}"
    subfolder_path = os.path.join(input_dir, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # Chọn các ảnh cho thư mục con này (100 ảnh mỗi thư mục)
    start_idx = i * images_per_folder
    end_idx = min(start_idx + images_per_folder, total_images)
    subfolder_images = image_files[start_idx:end_idx]

    # Di chuyển ảnh vào thư mục con
    for img_name in subfolder_images:
        src_path = os.path.join(input_dir, img_name)
        dst_path = os.path.join(subfolder_path, img_name)
        shutil.move(src_path, dst_path)
        print(f"Di chuyển {img_name} vào {subfolder_name}")

print("Đã tách thư mục thành công!")