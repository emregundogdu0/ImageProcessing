from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data="C:/Users/Emre Gundogdu/Desktop/fulldata/config.yaml",
    epochs=200,
    imgsz=640,
    device="cpu",
    name="mytrain",
    exist_ok=True,
    lr0=0.005,
    batch=8,
    mosaic=True,
    hsv_h=0.015,
    translate=0.1,
    scale=0.5,
    degrees=5
)

trained_model = YOLO("runs/detect/mytrain/weights/best.pt")

predict_results = trained_model(
    source="C:/Users/Emre Gundogdu/Desktop/tumfoto",
    save=True
)
