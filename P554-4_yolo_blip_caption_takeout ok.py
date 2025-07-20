# detect pen,pencil box,eraser,  inside outside,
# when hand on,capture image,display inference result/ voice in/out
import multiprocessing
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO

# ========== 配置 ==========
yolo_model_path = "runs/detect/yolov8_custom20/weights/best.pt"
blip_model_path = "/home/eric/PycharmProjects/PythonProject2/multimodal_project/data_for_blip_train2/checkpoint-90"
camera_index = 2
conf_threshold = 0.5
speak_interval = 8  # 同一个内容播报最短间隔（秒）
HAND_WINDOW_NAME = "Hand Detected"

# 状态缓存与阈值设置
pen_status_log = []  # [{'time': timestamp, 'inside': True/False}]
check_interval = 2  # 秒


# ========== 语音播报子进程 ==========
def speaker_process(q):
    while True:
        text = q.get()
        if text is None:
            break
        os.system(f'espeak "{text}" >/dev/null 2>&1')


speak_q = multiprocessing.Queue()
speak_proc = multiprocessing.Process(target=speaker_process, args=(speak_q,))
speak_proc.start()

# ========== 模型加载 ==========
print("\U0001F4E6 加载 YOLOv8 模型...")
yolo_model = YOLO(yolo_model_path)

print("\U0001F9E0 加载 BLIP 模型（处理器使用预训练）...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_path).to(
    "cuda" if torch.cuda.is_available() else "cpu")

# ========== 摄像头初始化 ==========
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

print("\U0001F3A5 正在运行：按 'q' 键退出")
font = cv2.FONT_HERSHEY_SIMPLEX
last_spoken = {}
last_check_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    results = yolo_model.predict(source=frame, conf=conf_threshold, verbose=False)
    boxes = results[0].boxes

    pen_inside = False
    pen_box = None
    pencil_box = None

    if boxes is not None and len(boxes.cls) > 0:
        for i in range(len(boxes.cls)):
            cls_id = int(boxes.cls[i])
            class_name = yolo_model.names[cls_id]
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
            label = f"{class_name} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), font, 0.6, (255, 255, 0), 2)

            if class_name in ["blue_pen", "red_pen", "black_pen", "blue_marker_pen"]:
                pen_box = (x1, y1, x2, y2)
            elif class_name == "pencil_box":
                pencil_box = (x1, y1, x2, y2)
            elif class_name == "hand":
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                if x2 > x1 and y2 > y1:
                    hand_crop = frame[y1:y2, x1:x2].copy()
                    gray_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
                    brightness = np.mean(gray_crop)

                    if brightness > 10:
                        try:
                            pil_image = Image.fromarray(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
                            inputs = blip_processor(images=pil_image, return_tensors="pt").to(blip_model.device)
                            out = blip_model.generate(**inputs, max_new_tokens=30)
                            caption = blip_processor.decode(out[0], skip_special_tokens=True)
                        except Exception as e:
                            caption = "BLIP 推理失败"
                            print("推理失败：", str(e))

                        cv2.putText(frame, caption, (x1, y2 + 25), font, 0.6, (0, 255, 255), 2)

                        caption_img = np.ones((200, 1000, 3), dtype=np.uint8) * 255

                        line_y = 40
                        line_gap = 35  # 行间距

                        # Caption（蓝色）
                        cv2.putText(caption_img, f"{caption}", (30, line_y), font, 0.8, (255, 0, 0), 2)

                        # Pen 位置（红色或绿色）
                        line_y += line_gap
                        pen_pos_str = f"Pen: {pen_box}" if pen_box else "Pen: Not detected"
                        pen_color = (0, 180, 0) if pen_box else (0, 0, 255)
                        cv2.putText(caption_img, pen_pos_str, (30, line_y), font, 0.6, pen_color, 1)

                        # Pencil Box 位置（绿色）
                        line_y += line_gap
                        pencil_box_str = f"Pencil Box: {pencil_box}" if pencil_box else "Pencil Box: Not detected"
                        cv2.putText(caption_img, pencil_box_str, (30, line_y), font, 0.6, (0, 200, 0), 1)

                        cv2.imshow("Caption Text", caption_img)

                        cv2.putText(caption_img, caption, (30, 120), font, 1, (255, 0, 0), 2)
                        cv2.imshow("Caption Text", caption_img)

                        if now - last_spoken.get("caption", 0) > speak_interval:
                            speak_q.put(caption)
                            last_spoken["caption"] = now

                        resized = cv2.resize(hand_crop, (0, 0), fx=0.5, fy=0.5)
                        cv2.imshow(HAND_WINDOW_NAME, resized)
                        print("📢 Caption:", caption)

        if pen_box and pencil_box:
            px = (pen_box[0] + pen_box[2]) // 2
            py = (pen_box[1] + pen_box[3]) // 2
            if pencil_box[0] < px < pencil_box[2] and pencil_box[1] < py < pencil_box[3]:
                pen_inside = True

    if now - last_check_time >= check_interval:
        pen_status_log.append({'time': now, 'inside': pen_inside})

        if len(pen_status_log) >= 2:
            prev = pen_status_log[-2]['inside']
            curr = pen_status_log[-1]['inside']

            if prev and not curr:
                action_caption = "The pen was taken out of the pencil box."
                speak_q.put(action_caption)
            elif not prev and curr:
                action_caption = "The pen was put into the pencil box."
                speak_q.put(action_caption)
            else:
                action_caption = None

            if action_caption:
                caption_img = np.ones((200, 1000, 3), dtype=np.uint8) * 255
                cv2.putText(caption_img, action_caption, (30, 120), font, 1, (255, 0, 0), 2)
                cv2.imshow("Caption Text", caption_img)
                print("🧠 推理：", action_caption)

        last_check_time = now

    cv2.imshow("YOLO + BLIP Caption", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
speak_q.put(None)
speak_proc.join()
print("已退出程序")
