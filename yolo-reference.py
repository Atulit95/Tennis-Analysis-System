from ultralytics import YOLO

model=YOLO("yolov8x")

results=model.predict("inputs/input_image1.jpg",save=True)

print(results)
print("boxes:")
for box in results[0].boxes:
    print(box)