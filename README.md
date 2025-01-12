Phát hiện ổ gà sử dụng Faster R-CNN

Dự án này sử dụng mô hình Faster R-CNN để phát hiện ổ gà trong hình ảnh đường phố. Bộ dữ liệu bao gồm các hình ảnh được gắn nhãn chỉ ra sự hiện diện của ổ gà. README này sẽ hướng dẫn bạn qua cấu trúc dự án, cài đặt và hướng dẫn sử dụng.

#Cấu trúc Dự án

.
├── config.py            # Cài đặt cấu hình cho mô hình và quá trình huấn luyện
├── dataset.py           # Xử lý dữ liệu và định nghĩa bộ tải dữ liệu
├── engine.py            # Động cơ huấn luyện và các hàm liên quan
├── model.py             # Tạo mô hình sử dụng Faster R-CNN
├── test.py              # Script kiểm thử mô hình
├── train.py             # Script huấn luyện mô hình
├── utils.py             # Các hàm tiện ích cho biến đổi, trực quan hóa, v.v.
├── train_df.csv         # Tệp CSV chứa dữ liệu gán nhãn cho huấn luyện
├── road-pothole-images  # Thư mục chứa bộ dữ liệu
└── output               # Thư mục lưu các tệp đầu ra

Cài đặt

Yêu cầu

Python 3.7+

PyTorch

torchvision

albumentations

OpenCV

pandas

numpy

matplotlib

tqdm

Cài đặt

Clone kho lưu trữ:

git clone https://github.com/your-repo/pothole-detection.git
cd pothole-detection

Cài đặt các gói Python cần thiết:

pip install -r requirements.txt

Cấu hình

Chỉnh sửa config.py để thiết lập các tham số huấn luyện và đường dẫn đến bộ dữ liệu:

BACKBONE: Mô hình backbone cho Faster R-CNN (resnet50, mobilenet_v3_large, v.v.).

BATCH_SIZE: Kích thước batch cho huấn luyện.

NUM_EPOCHS: Số lượng epoch huấn luyện.

DEVICE: Thiết bị để chạy mô hình (CPU hoặc CUDA).

TRAIN_DIR và TEST_DIR: Đường dẫn đến bộ dữ liệu huấn luyện và kiểm thử.

CLASSES: Danh sách nhãn lớp.

LEARNING_RATE: Tốc độ học cho optimizer.

MIN_SIZE: Kích thước hình ảnh tối thiểu cho mô hình.

PREDICTION_THRES: Ngưỡng lọc dự đoán trong quá trình kiểm thử.

OUT_DIR: Thư mục lưu các tệp đầu ra (mô hình, biểu đồ, v.v.).

Sử dụng

Huấn luyện

Chạy script huấn luyện:

python train.py

Điều này sẽ huấn luyện mô hình và lưu biểu đồ mất mát cũng như các checkpoint mô hình theo khoảng thời gian được chỉ định.

Để trực quan hóa hình ảnh đã biến đổi trong quá trình huấn luyện, đặt VISUALIZE_TRANSFORMED_IMAGES thành True trong config.py và chạy:

python dataset.py

Kiểm thử

Sau khi huấn luyện, chạy script kiểm thử để đánh giá mô hình:

python test.py

Điều này sẽ xử lý các hình ảnh kiểm thử, vẽ các hộp bao quanh các ổ gà được phát hiện và lưu các hình ảnh đầu ra vào thư mục test_prediction.

Lời cảm ơn

Bộ dữ liệu sử dụng trong dự án này được lấy từ road-pothole-images-for-pothole-detection.

Việc triển khai dựa trên PyTorch và torchvision.

Giấy phép

Dự án này được cấp phép dưới giấy phép MIT - xem tệp LICENSE để biết chi tiết.

Đóng góp

Các đóng góp được hoan nghênh! Vui lòng gửi một pull request hoặc mở một issue cho bất kỳ lỗi hoặc yêu cầu tính năng nào.

Liên hệ

Để có bất kỳ câu hỏi hoặc thắc mắc nào, vui lòng liên hệ Tên của bạn.

README này cung cấp cái nhìn tổng quan về dự án Phát hiện ổ gà, bao gồm cài đặt, sử dụng và cấu trúc các tệp dự án. Hãy làm theo hướng dẫn cẩn thận để huấn luyện và kiểm thử mô hình phát hiện ổ gà.

