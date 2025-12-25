# src/preprocess.py
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
from feature_extractor import MediaPipeHandExtractor
from config import KEYPOINT_DIR, RAW_DIR, SEQUENCE_LEN

def process_video_file(video_path: Path, out_path: Path, seq_len=SEQUENCE_LEN, start_frame=None, end_frame=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_idx = 0
    extractor = MediaPipeHandExtractor()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if start_frame is not None and frame_idx < start_frame:
            frame_idx += 1
            continue
        if end_frame is not None and frame_idx > end_frame:
            break
        kp = extractor.extract_keypoints(frame)  # flat vector
        frames.append(kp)
        frame_idx += 1
    cap.release()
    extractor.close()

    if len(frames) == 0:
        # fallback: save one zero frame
        frames = [np.zeros_like(kp)]

    kps = np.stack(frames, axis=0).astype(np.float32)  # (T, D)
    kps = pad_or_trim(kps, seq_len)
    np.savez_compressed(out_path, kp=kps)
    return out_path

def process_frame_folder(frames_dir: Path, out_path: Path, seq_len=SEQUENCE_LEN):
    frame_files = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in [".jpg",".png",".jpeg"]])
    extractor = MediaPipeHandExtractor()
    frames = []
    for f in frame_files:
        img = cv2.imread(str(f))
        kp = extractor.extract_keypoints(img)
        frames.append(kp)
    extractor.close()
    if len(frames) == 0:
        frames = [np.zeros((2*21*3,), dtype=np.float32)]
    kps = np.stack(frames, axis=0).astype(np.float32)
    kps = pad_or_trim(kps, seq_len)
    np.savez_compressed(out_path, kp=kps)
    return out_path

def pad_or_trim(kps: np.ndarray, seq_len: int):
    T, D = kps.shape
    if T >= seq_len:
        return kps[:seq_len]
    else:
        pad = np.zeros((seq_len - T, D), dtype=np.float32)
        return np.vstack([kps, pad])

def process_manifest(manifest_csv: Path, seq_len=SEQUENCE_LEN, out_dir: Path = KEYPOINT_DIR):
    df = pd.read_csv(manifest_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        id_ = str(row['id'])
        video_path = Path(row['video_path'])
        start = int(row['start_frame']) if 'start_frame' in row and not pd.isna(row['start_frame']) else None
        end = int(row['end_frame']) if 'end_frame' in row and not pd.isna(row['end_frame']) else None
        out_path = out_dir / f"{id_}.npz"
        try:
            if video_path.is_dir():
                process_frame_folder(video_path, out_path, seq_len=seq_len)
            else:
                process_video_file(video_path, out_path, seq_len=seq_len, start_frame=start, end_frame=end)
        except Exception as e:
            print(f"[ERROR] id={id_} path={video_path} -> {e}")
            # write a zeroed sequence so dataset remains consistent
            D = 2 * 21 * 3
            np.savez_compressed(out_path, kp=np.zeros((seq_len, D), dtype=np.float32))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=str(Path(__file__).resolve().parent.parent / "data" / "labels.csv"))
    parser.add_argument("--seq_len", type=int, default=SEQUENCE_LEN)
    args = parser.parse_args()
    process_manifest(Path(args.manifest), seq_len=args.seq_len)
