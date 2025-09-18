from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="data_set/data.yaml",
    epochs=30,          # tăng nhẹ epoch (15 có thể hơi ít, 30–50 hợp lý hơn)
    imgsz=640,          # giữ nguyên, YOLO mặc định
    batch=2,            # giảm batch vì dataset nhỏ, giúp cập nhật gradient mượt hơn
    name="text_detector_small",
    plots=True,         # xuất biểu đồ loss, mAP, P, R
    patience=10         # early stopping: dừng nếu 10 epoch liên tiếp không cải thiện
)


metrics = model.val()
print(metrics)  # sẽ in Precision, Recall, mAP50, mAP50-95

