import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
CONFIDENCE_THRESH = 0.5   # keep detections above this score
MASK_THRESH       = 0.5   # binarize masks at this threshold
DOWNSCALE_FACTOR  = 1.0   # set <1.0 to speed up (at cost of resolution)
# ────────────────────────────────────────────────────────────────────────────────

def load_model(device):
    """Load pretrained Mask R‑CNN from torchvision."""
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval().to(device)
    return model

def preprocess_frame(bgr_frame, device):
    """Resize, convert BGR→RGB tensor, and send to device."""
    if DOWNSCALE_FACTOR != 1.0:
        h, w = bgr_frame.shape[:2]
        bgr_frame = cv2.resize(
            bgr_frame,
            (int(w * DOWNSCALE_FACTOR), int(h * DOWNSCALE_FACTOR)),
            interpolation=cv2.INTER_LINEAR
        )
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    tensor = transforms.ToTensor()(rgb).to(device)
    return bgr_frame, tensor

def compute_orientation(mask):
    """
    From a binary mask, fit a min‑area rotated rectangle and return:
      - box: 4×2 int coords of its corners
      - angle: in [0,180) degrees, relative to horizontal
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)            # ((cx,cy), (w,h), angle)
    box = cv2.boxPoints(rect).astype(int)
    angle = rect[2]

    # OpenCV angle is in [-90,0); convert to [0,180)
    w, h = rect[1]
    if w < h:
        angle = 90 + angle
    else:
        angle = -angle

    return box, angle

def run_webcam_orientation():
    # Device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(device)

    cap = cv2.VideoCapture(0)   # default webcam
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    print("📐 Object Orientation – Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break

        # Preprocess
        vis_frame, tensor = preprocess_frame(frame, device)

        # Inference
        with torch.no_grad():
            out = model([tensor])[0]

        scores = out["scores"].cpu().numpy()
        keep   = scores >= CONFIDENCE_THRESH
        boxes  = out["boxes"].cpu().numpy()[keep]
        masks  = out["masks"].cpu().numpy()[keep]  # (N,1,H,W)

        # Draw each detected object
        for box, mask_prob in zip(boxes, masks):
            # Binarize mask
            mask = (mask_prob[0] > MASK_THRESH).astype(np.uint8) * 255

            # Compute rotated box & angle
            rot_box, angle = compute_orientation(mask)
            if rot_box is None:
                continue

            # Draw rotated rect
            cv2.drawContours(vis_frame, [rot_box], -1, (0, 255, 0), 2)

            # Optionally draw axis-aligned bbox
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

            # Put angle label
            label = f"{angle:.1f}°"
            lx, ly = rot_box[0]
            cv2.putText(
                vis_frame, label,
                (lx, ly - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,               # font scale
                (0, 255, 0),       # color
                2,                 # thickness
                cv2.LINE_AA
            )

        # Show result
        cv2.imshow("Object Orientation (q to quit)", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_orientation()