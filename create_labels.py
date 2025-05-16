import os

# Đường dẫn đến thư mục chứa ảnh
input_dir = "ocr_data/input_images/subfolder_2"

# Đường dẫn đến tệp labels.txt sẽ được tạo
label_file = "labels.txt"

# Đảm bảo thư mục chứa ảnh tồn tại
if not os.path.exists(input_dir):
    print(f"Thư mục {input_dir} không tồn tại. Vui lòng kiểm tra lại!")
    exit()

# Tạo hoặc mở tệp labels.txt để ghi
with open(label_file, "w", encoding="utf-8") as f:
    # Lặp qua tất cả các tệp trong thư mục ảnh
    for img_name in os.listdir(input_dir):
        # Chỉ xử lý các tệp ảnh (jpg, png)
        if img_name.endswith(('.jpg', '.png')):
            # Tách phần văn bản thực tế từ tên tệp (phần trước dấu "_")
            ground_truth = img_name.split("_")[0]
            # Ghi dòng vào labels.txt với định dạng: tên_file.jpg    văn bản
            f.write(f"{img_name}\t{ground_truth}\n")

print(f"Tệp {label_file} đã được tạo thành công!")