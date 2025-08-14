# YOLOv8 & v11
We can each use it to suit the needs of the user

## Installation
<pre> <code> pip install ultralytics </code> </pre>

## YOLOv8 command
### using Webcam 
<pre> <code> python3 infer.py --yolo v8 --weights weights/yolov8n.pt --conf 0.25 --source 0 --view-img </code> </pre>

### using Video
<pre> <code> python3 infer.py --yolo v8 --weights weights/yolov8n.pt --conf 0.25 --source inputs/drive.mp4 --view-img </code> </pre>
![Image](https://github.com/user-attachments/assets/60ae89ef-3a48-4bb4-b8bd-498858b9877e)

## YOLOv11 command
### using Webcam
<pre> <code> python3 infer.py --yolo v11 --weights weights/yolo11n.pt --conf 0.25 --source 0 --view-img </code> </pre>

### using Video
<pre> <code> python3 infer.py --yolo v11 --weights weights/yolo11n.pt --conf 0.25 --source inputs/drive.mp4 --view-img </code> </pre>
![Image](https://github.com/user-attachments/assets/3caae793-fe3e-4b5e-942c-b7f3ed00585d)
