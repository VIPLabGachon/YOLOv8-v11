# YOLOv8 & v11
We can each use it to suit the needs of the user

## Installation
<pre> <code> pip install ultralytics </code> </pre>

## YOLOv8 command
### Webcam 사용
<pre> <code> python3 infer.py --yolo v8 --weights weights/yolov8n.pt --conf 0.25 --source 0 --view-img </code> </pre>

### Video 사용
<pre> <code> python3 infer.py --yolo v8 --weights weights/yolov8n.pt --conf 0.25 --source inputs/drive.mp4 --view-img </code> </pre>

## YOLOv11 command
### Webcam 사용
<pre> <code> python3 infer.py --yolo v11 --weights weights/yolo11n.pt --conf 0.25 --source 0 --view-img </code> </pre>

### Video 사용
<pre> <code> python3 infer.py --yolo v11 --weights weights/yolo11n.pt --conf 0.25 --source inputs/drive.mp4 --view-img </code> </pre>
