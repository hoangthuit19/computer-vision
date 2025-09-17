from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="data_set/data.yaml",
    epochs=5,         # giảm số epoch
    imgsz=640,
    batch=4,           # batch nhỏ vì data ít
    name="text_detector_small",
    plots=True         # xuất biểu đồ loss, mAP, P, R
)

metrics = model.val()
print(metrics)  # sẽ in Precision, Recall, mAP50, mAP50-95

