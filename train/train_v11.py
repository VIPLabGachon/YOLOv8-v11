from ultralytics import YOLO

model = YOLO('yolov11.yaml')
model.train(data='data.yaml', epochs=100, imgsz=640, batch=16, name='yolov11')
model.export(format='torchscript')