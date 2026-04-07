import torch
import cv2
import os
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

# تحديد مسار المشروع
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "video_classification_model")

# تحميل الموديل
processor = VideoMAEImageProcessor.from_pretrained(MODEL_PATH)
model = VideoMAEForVideoClassification.from_pretrained(
    MODEL_PATH,
    ignore_mismatched_sizes=True
)

model.eval()


def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

    cap.release()
    return frames


def predict_video(video_path):
    frames = extract_frames(video_path)

    if len(frames) == 0:
        return "Error"

    inputs = processor(frames, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    pred = torch.argmax(logits).item()

    
    if pred == 1:
        return "Theft"
    else:
        return "Not Theft"