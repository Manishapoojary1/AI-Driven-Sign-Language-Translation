import cv2, os
from pathlib import Path

# where captured samples will go
out_base = Path("data/webcam_samples")
out_base.mkdir(parents=True, exist_ok=True)

print("Enter labels to capture (e.g. A,B,C): ")
labels = input("Labels: ").strip().split(",")
labels = [l.strip().upper() for l in labels if l.strip()]

for lbl in labels:
    (out_base / lbl).mkdir(parents=True, exist_ok=True)

print("Controls: SPACE = save frame, N = next label, P = previous label, Q = quit")

cap = cv2.VideoCapture(0)
idx = 0
counts = {l: len(list((out_base / l).glob('*.jpg'))) for l in labels}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    label = labels[idx]
    cv2.putText(frame, f"Label: {label}  Count: {counts[label]}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # spacebar to save
        fname = out_base / label / f"{counts[label]+1:04d}.jpg"
        cv2.imwrite(str(fname), frame)
        counts[label] += 1
        print("Saved", fname)
    elif key == ord('n'):
        idx = (idx + 1) % len(labels)
    elif key == ord('p'):
        idx = (idx - 1) % len(labels)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
