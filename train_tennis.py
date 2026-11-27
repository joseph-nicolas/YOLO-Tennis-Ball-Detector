"""Train the custom tennis ball detection model with Ultralytics YOLO11."""

from ultralytics import YOLO


# Load the pretrained YOLO11n model. This downloads yolo11n.pt the first
# I ran it, and then it just reuses what's already there.
model = YOLO("yolo11n.pt")

# Train it on our custom dataset for 100 epochs at 1280x1280. Took about
# 3 hours 30 minutes on my Laptop RTX 4060.
results = model.train(
    data="C:/Users/joseph/Projects/Tennis_Dataset/data.yaml",
    epochs=100,
    imgsz=1280,
)

# After it is finished it will export the metrics, results and best.pt weights
