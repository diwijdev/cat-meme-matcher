def match_meme(features: dict, base_brow=None):
    mo = features["mouth_open"]
    eo = features["eye_open"]
    mw = features["mouth_w"]
    su = features["smile_up"]
    br = features.get("brow_raise")

    # Angry override (brows down)
    if base_brow is not None and br is not None:
        brow_drop = base_brow - br
        if brow_drop > 0.0023 and mo < 0.030 and su < 0.014:
            conf = min(0.95, 0.6 + (brow_drop - 0.0023) / 0.003)
            return "angry", float(conf)

    # Mode select
    if mo > 0.060:
        mode = "open_mouth"
    elif mw > 0.260:
        mode = "smile_width"
    elif eo < 0.030:
        mode = "squint"
    else:
        mode = "default"

    if mode == "open_mouth":
        if eo < 0.030 and su > 0.010:
            return "laughing", 0.90
        if mo > 0.085 and eo > 0.050:
            return "shocked", 0.85
        if 0.030 <= mo <= 0.085 and eo > 0.055:
            return "scared", 0.75
        return "shocked", 0.55

    if mode == "squint":
        return "judging", 0.80

    if mode == "smile_width":
        if mw >= 0.285 and su > 0.015:
            return "smiling", 0.85
        if mw >= 0.265:
            return "polite", 0.70
        return "staring", 0.40


    return "staring", 0.40
