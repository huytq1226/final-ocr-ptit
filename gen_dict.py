import random

base_vocab = [
    # Địa danh
    "Hà Nội", "Sài Gòn", "Huế", "Đà Nẵng", "Cần Thơ", "Hội An", "Phú Quốc", "Nha Trang", "Đà Lạt", "Yên Tử",
    "Phong Nha", "Vịnh Hạ Long", "Sa Pa", "Bến Nhà Rồng", "Chợ Bến Thành", "Chùa Một Cột", "Dinh Độc Lập",
    "Cầu Rồng", "Lăng Bác", "Mũi Né",

    # Đồ ăn / Ẩm thực
    "Phở bò", "Bánh mì", "Bánh chưng", "Bánh tét", "Cà phê sữa đá", "Mứt Tết", "Nước mắm", "Bún chả", "Chả giò",
    "Bánh cuốn", "Gỏi cuốn", "Cơm tấm", "Hủ tiếu", "Bánh bèo", "Bánh xèo", "Trà sữa", "Chè đậu xanh", "Cháo lòng",

    # Cảm xúc / Trạng thái
    "Hạnh phúc", "Buồn bã", "Tức giận", "Bình yên", "Tự do", "Yêu thương", "Sợ hãi", "Tò mò", "Tự tin", "Đam mê",

    # Công nghệ / Khoa học
    "Máy tính", "Điện thoại", "Trí tuệ nhân tạo", "Học máy", "Lập trình", "Phần mềm", "Kỹ thuật số", "Mạng xã hội",
    "Blockchain", "Internet vạn vật", "Dữ liệu lớn", "Máy học", "Thị giác máy", "Nhận dạng ký tự", "Robot thông minh",

    # Nghề nghiệp
    "Giáo viên", "Kỹ sư", "Bác sĩ", "Nông dân", "Lập trình viên", "Nhà khoa học", "Nhà nghiên cứu", "Thiết kế đồ họa",
    "Người mẫu", "Nghệ sĩ", "Ca sĩ", "Diễn viên", "Tài xế", "Nhà báo", "Nhiếp ảnh gia",

    # Chủ đề xã hội / Văn hóa
    "Gia đình", "Bạn bè", "Tình yêu", "Hòa bình", "Đoàn kết", "Văn hóa", "Truyền thống", "Lịch sử", "Tổ quốc", "Đồng bào",
    "Môi trường", "Biến đổi khí hậu", "Phát triển bền vững", "Năng lượng sạch", "Tái chế", "Ô nhiễm không khí",

    # Thể thao
    "Bóng đá", "Bóng chuyền", "Cầu lông", "Chạy bộ", "Đạp xe", "Bơi lội", "Bóng bàn", "Tennis", "Cờ vua",

    # Thiên nhiên / Địa lý
    "Biển Đông", "Sông Mekong", "Đồng bằng", "Cao nguyên", "Núi rừng", "Hang động", "Rừng ngập mặn", "Hồ nước ngọt",

    # Học đường
    "Trường học", "Sinh viên", "Giảng viên", "Giáo trình", "Kỳ thi", "Thư viện", "Lớp học", "Bài tập", "Giờ ra chơi",

    # Dịp lễ / Tết
    "Tết Nguyên Đán", "Trung thu", "Lì xì", "Câu đối đỏ", "Mâm ngũ quả", "Lễ hội", "Hoa đào", "Hoa mai", "Trăng Rằm"
]

# Thêm từ đơn giản, phổ biến
simple_words = ["xin", "chào", "việt", "nam", "cong", "nghe", "tri", "tue", "nhan", "tao", "hoa", "dao", "tet", "trung", "thu", "phat", "trien"]

# Kết hợp thêm các cụm ngẫu nhiên để đủ 1000
additional = set()
while len(additional) + len(base_vocab) < 1000:
    new_phrase = " ".join(random.sample(simple_words, k=random.randint(1, 3))).capitalize()
    additional.add(new_phrase)

all_phrases = list(set(base_vocab) | additional)
random.shuffle(all_phrases)

with open("dict_vi.txt", "w", encoding="utf-8") as f:
    for line in all_phrases[:1000]:
        f.write(line + "\n")
