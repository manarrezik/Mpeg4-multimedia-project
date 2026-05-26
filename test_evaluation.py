# test_evaluation.py
import cv2
import numpy as np
from preprocessing  import preprocess_frame, reconstruct_frame
from intra_coding   import encode_iframe, decode_iframe
from inter_coding   import encode_pframe, decode_pframe
from entropy_coding import encode_to_bin, decode_from_bin
from evaluation     import compute_psnr, print_metrics, visualize_pipeline

# ── Frames ──────────────────────────────────────────────────────────────────
frame1 = cv2.imread("frames/Rectangle 16.png")
frame2 = cv2.GaussianBlur(frame1, (5, 5), 0)
frame3 = cv2.GaussianBlur(frame1, (7, 7), 0)
frame4 = cv2.GaussianBlur(frame1, (9, 9), 0)
original_frames = [frame1, frame2, frame3, frame4]

GOP_SIZE = 2
QUALITY  = 50

encoded_frames       = []
reconstructed_Y      = []
reconstructed_frames = []
frame_types          = []
ref_Y                = None

# ── Encode ───────────────────────────────────────────────────────────────────
for idx, frame in enumerate(original_frames):
    Y, Cb, Cr = preprocess_frame(frame)

    if idx % GOP_SIZE == 0:
        enc   = encode_iframe(Y, Cb, Cr, quality=QUALITY)
        Y_rec, Cb_rec, Cr_rec = decode_iframe(enc)
        ftype = 'I'
    else:
        enc   = encode_pframe(Y, ref_Y, quality=QUALITY)
        Y_rec = decode_pframe(enc, ref_Y)
        Cb_rec, Cr_rec = Cb, Cr
        ftype = 'P'

    ref_Y = Y_rec
    encoded_frames.append(enc)
    reconstructed_Y.append(Y_rec)
    frame_types.append(ftype)

    rec_bgr = reconstruct_frame(Y_rec, Cb_rec, Cr_rec)
    reconstructed_frames.append(rec_bgr)
    print(f"Frame {idx} → {ftype}-frame encoded ✅")

# ── Save .bin ────────────────────────────────────────────────────────────────
encode_to_bin(encoded_frames, "output/compressed.bin")

# ── PSNR ─────────────────────────────────────────────────────────────────────
psnr_values = [
    compute_psnr(original_frames[i], reconstructed_frames[i])
    for i in range(len(original_frames))
]

# ── Metrics ───────────────────────────────────────────────────────────────────
print_metrics(frame_types, psnr_values, original_frames, "output/compressed.bin")

# ── Visualise ─────────────────────────────────────────────────────────────────
visualize_pipeline(
    original_frames,
    reconstructed_Y,
    encoded_frames,
    frame_types,
    psnr_values,
    "output/compressed.bin"
)