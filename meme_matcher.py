import time
from collections import Counter, deque

import cv2
import numpy as np
import mediapipe as mp
import os


MEME_DIR = "memes"
MEME_FILES = {
    "shocked": "OMG-Cat.jpg",
    "judging": "suscat.png",
    "laughing": "laughing-cat.jpg",
    "polite": "polite cat.jpg",
    "scared": "Scared-Cat.jpg",
    "smiling": "Smiling-Cat.jpg",
    "staring": "staring-cat.jpg",
    "angry": "angrycat.png",
}

def load_memes():
    memes = {}
    for k, fname in MEME_FILES.items():
        path = os.path.join(MEME_DIR, fname)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not load meme image: {path}")
        memes[k] = img
    return memes


# -----------------------------
# Math helpers
# -----------------------------
def dist(a, b):
    return float(np.linalg.norm(a - b))

def safe_div(a, b, eps=1e-6):
    return float(a / (b + eps))

# Landmark indices
IDX = {
    "upper_lip": 13,
    "lower_lip": 14,
    "mouth_left": 61,
    "mouth_right": 291,
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "right_eye_top": 386,
    "right_eye_bottom": 374,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_brow": 105,
    "right_brow": 334,
    "forehead": 10,
    "chin": 152,
    "nose_tip": 1,
}

def unit(v, eps=1e-6):
    n = float(np.linalg.norm(v))
    return v / (n + eps)

def unit3(v, eps=1e-8):
    n = float(np.linalg.norm(v))
    return v / (n + eps)

def proj3(v, axis_unit):
    return float(np.dot(v, axis_unit))


def perp2(v):
    # rotate 2D vector 90 degrees
    return np.array([-v[1], v[0]], dtype=np.float32)

def proj(a, axis_unit):
    # scalar projection of vector a onto axis_unit
    return float(np.dot(a, axis_unit))


def extract_features_3d(pts2, pts3):
    # Scale for normalization (use face height in 2D pixels)
    face_h = dist(pts2[IDX["forehead"]], pts2[IDX["chin"]])

    # --- 2D features (stable enough) ---
    upper2 = pts2[IDX["upper_lip"]]
    lower2 = pts2[IDX["lower_lip"]]
    ml2 = pts2[IDX["mouth_left"]]
    mr2 = pts2[IDX["mouth_right"]]

    mouth_open = safe_div(dist(upper2, lower2), face_h)
    mouth_w = safe_div(dist(ml2, mr2), face_h)

    eye_open = safe_div(
        (
            dist(pts2[IDX["left_eye_top"]], pts2[IDX["left_eye_bottom"]]) +
            dist(pts2[IDX["right_eye_top"]], pts2[IDX["right_eye_bottom"]])
        ) / 2,
        face_h,
    )

    # --- Build a 3D face frame in MediaPipe landmark space ---
    le3 = pts3[IDX["left_eye_outer"]]
    re3 = pts3[IDX["right_eye_outer"]]
    nose3 = pts3[IDX["nose_tip"]]

    origin = (le3 + re3) / 2.0

    x_axis = unit3(re3 - le3)             # left->right
    forward_hint = unit3(nose3 - origin)  # points roughly "forward/down" depending on coords

    # Make y_axis orthogonal to x_axis using Gram-Schmidt
    # y starts as forward_hint with x component removed
    y_axis = forward_hint - proj3(forward_hint, x_axis) * x_axis
    y_axis = unit3(y_axis)

    z_axis = unit3(np.cross(x_axis, y_axis))  # completes right-handed frame

    # --- 3D smile_up: mouth corners vs mouth center along face Y ---
    upper3 = pts3[IDX["upper_lip"]]
    lower3 = pts3[IDX["lower_lip"]]
    ml3 = pts3[IDX["mouth_left"]]
    mr3 = pts3[IDX["mouth_right"]]

    mouth_center3 = (upper3 + lower3) / 2.0
    corners_avg3 = (ml3 + mr3) / 2.0

    v_smile = corners_avg3 - mouth_center3
    smile_up = -proj3(v_smile, y_axis)    # negative if y points down-ish; this makes "up" positive

    # --- Optional brow_raise in face frame (if you added brow indices) ---
    brow_raise = None
    if "left_brow" in IDX and "right_brow" in IDX:
        lb3 = pts3[IDX["left_brow"]]
        rb3 = pts3[IDX["right_brow"]]
        le_top3 = pts3[IDX["left_eye_top"]]
        re_top3 = pts3[IDX["right_eye_top"]]

        vL = lb3 - le_top3
        vR = rb3 - re_top3
        brow_raise = - (proj3(vL, y_axis) + proj3(vR, y_axis)) / 2.0

    out = {
        "mouth_open": float(mouth_open),
        "mouth_w": float(mouth_w),
        "eye_open": float(eye_open),

        # These are in normalized 3D landmark units; still consistent frame-to-frame
        "smile_up": float(smile_up),
    }
    if brow_raise is not None:
        out["brow_raise"] = float(brow_raise)

    return out



# Your baseline neutral readings (from your tests)
NEUTRAL = {
    "mouth_open": 0.001,
    "eye_open": 0.057,
    "mouth_w": 0.318,
}

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def band(x, lo, hi, softness=0.01):
    # ~1 inside [lo,hi], fades to 0 outside
    if x < lo:
        return clamp01((x - (lo - softness)) / softness)
    if x > hi:
        return clamp01(((hi + softness) - x) / softness)
    return 1.0

def match_meme_key(f, base_brow=None):
    mo = f["mouth_open"]
    eo = f["eye_open"]
    mw = f["mouth_w"]
    su = f["smile_up"]
    br = f.get("brow_raise", None)

    # --- Angry override: neutral face + brows down (no squint needed) ---
    if base_brow is not None and br is not None:
        brow_drop = base_brow - br  # positive when brows are lower than neutral

        # Your neutral ~0.016, angry ~0.013 => drop ~0.003
        if brow_drop > 0.0023 and mo < 0.030 and su < 0.014:
            conf = min(0.95, 0.6 + (brow_drop - 0.0023) / 0.003)
            return "angry", conf


    # -------------------------
    # Stage 1: pick a mode
    # -------------------------
    # Big mouth open dominates (expressions that involve open mouth)
    if mo > 0.060:
        mode = "open_mouth"
    # Very squinty dominates (but exclude open-mouth laughs)
    elif eo < 0.030:
        mode = "squint"
    # Wide-mouth smiles/polite
    elif mw > 0.340:
        mode = "smile_width"
    else:
        mode = "default"

    # -------------------------
    # Stage 2: decide within mode
    # -------------------------
    if mode == "open_mouth":
        # Laughing: open mouth + squint + some smile
        if eo < 0.030 and su > 0.010:
            return "laughing", 0.9
        # Shocked vs scared: mouth size + eyes
        if mo > 0.085 and eo > 0.050:
            return "shocked", 0.85
        if 0.030 <= mo <= 0.085 and eo > 0.055:
            return "scared", 0.75
        # fallback in this mode
        return "shocked", 0.55

    if mode == "squint":
        return "judging", 0.8

    if mode == "smile_width":
        # Your new desired behavior: polite starts >=0.36, smile starts >=0.40
        if mw >= 0.40 and su > 0.015:
            return "smiling", 0.85
        if mw >= 0.36:
            return "polite", 0.70
        return "staring", 0.4

    return "staring", 0.4








def main():
    model_path = "models/face_landmarker.task"

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
    )

    cap = cv2.VideoCapture(0)
    memes = load_memes()
    history = deque(maxlen=12)
    brow_samples = []
    BROW_CALIB_FRAMES = 30   # ~1 sec
    BASE_BROW = None



    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ts_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            label = "staring"
            features = None
            conf = 0.0
            if result.face_landmarks:
                lms = result.face_landmarks[0]
                pts2 = np.array([(lm.x * w, lm.y * h) for lm in lms], dtype=np.float32)

                # 3D in MediaPipe normalized coordinates:
                # x, y are normalized [0..1], z is relative (typically negative forward)
                pts3 = np.array([(lm.x, lm.y, lm.z) for lm in lms], dtype=np.float32)
                features = extract_features_3d(pts2, pts3)
                # ---- Calibrate neutral brow_raise for the first ~1 second ----
                if BASE_BROW is None:
                    brow_samples.append(features["brow_raise"])
                    if len(brow_samples) >= BROW_CALIB_FRAMES:
                        BASE_BROW = float(np.mean(brow_samples))
                        print("Calibrated BASE_BROW:", BASE_BROW)
                pred, conf = match_meme_key(features, BASE_BROW)
                history.append(pred)
                winner, count = Counter(history).most_common(1)[0]
                label = winner if count >= 6 else label  # 6 makes it easier to see new memes


                for key in IDX:
                    x, y = pts2[IDX[key]].astype(int)
                    cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)


            # === Overlay drawn at the very end (so nothing covers it) ===
            cv2.rectangle(frame, (10, 10), (520, 235), (0, 0, 0), -1)

            cv2.putText(frame, f"Match: {label}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if features is not None:
                cv2.putText(frame, f"mouth_open: {features['mouth_open']:.3f}", (20, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"eye_open:   {features['eye_open']:.3f}", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"mouth_w:    {features['mouth_w']:.3f}", (20, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"smile_up:  {features['smile_up']:.3f}", (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"brow_raise: {features['brow_raise']:.3f}", (20, 185),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                cv2.putText(frame, f"conf: {conf:.2f}", (20, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                if BASE_BROW is not None:
                    cv2.putText(frame, f"BASE_BROW:  {BASE_BROW:.3f}", (20, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)



            # --- Meme overlay (top-right) ---
            meme_key = label  # label should now be: shocked/judging/smug/neutral
            meme_img = memes.get(meme_key)

            if meme_img is not None:
                target_w = 220
                scale = target_w / meme_img.shape[1]
                target_h = int(meme_img.shape[0] * scale)
                meme_resized = cv2.resize(meme_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

                x1 = frame.shape[1] - target_w - 10
                y1 = 10
                x2 = x1 + target_w
                y2 = y1 + target_h

                # Keep inside frame bounds
                if y2 < frame.shape[0] and x1 >= 0:
                    frame[y1:y2, x1:x2] = meme_resized


            cv2.imshow("Cat Meme Matcher", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
