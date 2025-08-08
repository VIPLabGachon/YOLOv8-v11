import argparse
from ultralytics import YOLO
import os
import cv2
import time
from datetime import datetime

def run_inference(yolo_ver, weights, source, imgsz=640, conf=0.25, save=True, view_img=False, out_dir="runs/infer"):
    os.makedirs(out_dir, exist_ok=True)
    if not weights:
        weights = "yolov8n.pt" if yolo_ver == "v8" else "yolo11n.pt"
    model = YOLO(weights)

    is_video = str(source).endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')) or str(source).isdigit()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(out_dir, f"{yolo_ver}_{ts}.mp4")

    if is_video:
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open source: {source}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            t0 = time.time()
            results = model.predict(source=frame, imgsz=imgsz, conf=conf, save=False, verbose=False)
            ms = (time.time() - t0) * 1000.0
            fps_val = 1000.0 / ms if ms > 0 else 0.0
            annotated = results[0].plot()
            txt = f'{yolo_ver.upper()} | FPS: {fps_val:.2f} | {ms:.1f} ms'
            cv2.putText(annotated, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            out.write(annotated)
            if view_img:
                cv2.imshow("YOLO Inference", annotated)
                if cv2.waitKey(1) == 27:
                    break

        cap.release()
        out.release()
        if view_img:
            cv2.destroyAllWindows()
        if not save:
            if os.path.exists(out_path):
                os.remove(out_path)
    else:
        results = model.predict(source=source, imgsz=imgsz, conf=conf, save=save, stream=True, verbose=False)
        if view_img:
            for r in results:
                im = r.plot()
                cv2.imshow("YOLO Inference", im)
                if cv2.waitKey(1) == 27:
                    break
            cv2.destroyAllWindows()
        else:
            for _ in results:
                pass

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--yolo', type=str, choices=['v8', 'v11'], required=True, help='choose yolo version')
    p.add_argument('--weights', type=str, default='', help='path to weights; default depends on --yolo')
    p.add_argument('--source', type=str, required=True, help='image/folder/video/webcam (e.g., 0 or test.jpg)')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--nosave', action='store_true')
    p.add_argument('--view-img', action='store_true')
    p.add_argument('--out-dir', type=str, default='runs/infer')
    opt = p.parse_args()

    run_inference(
        yolo_ver=opt.yolo,
        weights=opt.weights,
        source=opt.source,
        imgsz=opt.imgsz,
        conf=opt.conf,
        save=not opt.nosave,
        view_img=opt.view_img,
        out_dir=opt.out_dir
    )
