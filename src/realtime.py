# src/realtime.py
import cv2
import numpy as np
import torch
from feature_extractor import MediaPipeHandExtractor
from models import SeqModel
from pathlib import Path
from config import DEVICE, SEQUENCE_LEN
import time

def load_model(ckpt_path, sample_dim):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    label_map = ckpt['label_map']
    num_classes = len(label_map)
    model = SeqModel(input_dim=sample_dim, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    inv_map = {v:k for k,v in label_map.items()}
    return model, inv_map

def run_webcam(ckpt_path):
    cap = cv2.VideoCapture(0)
    extractor = MediaPipeHandExtractor()
    # get sample dimension using a dummy image
    ret, frame = cap.read()
    if not ret:
        print("Cannot open webcam")
        return
    sample_kp = extractor.extract_keypoints(frame)
    sample_dim = sample_kp.shape[0]
    model, inv_map = load_model(ckpt_path, sample_dim)

    buffer = []
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        kp = extractor.extract_keypoints(frame)  # flat array
        buffer.append(kp)
        # keep LAST SEQUENCE_LEN frames
        if len(buffer) > SEQUENCE_LEN:
            buffer = buffer[-SEQUENCE_LEN:]
        # every N frames, run inference
        if len(buffer) == SEQUENCE_LEN:
            x = np.stack(buffer).astype('float32')  # (T,D)
            import torch
            xt = torch.from_numpy(x[None]).to(DEVICE)
            with torch.no_grad():
                logits = model(xt)
                pred = logits.argmax(1).item()
            label = inv_map[pred]
        else:
            label = "..."
        # overlay
        cv2.putText(frame, f"Pred: {label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("SignTranslate", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import fire
    fire.Fire(run_webcam)
