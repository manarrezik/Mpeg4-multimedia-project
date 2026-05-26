# preprocessing.py
import numpy as np
import cv2

def bgr_to_ycbcr(frame_bgr):
    """Convert BGR frame to YCbCr color space."""
    frame_float = frame_bgr.astype(np.float32)

    B = frame_float[:, :, 0]
    G = frame_float[:, :, 1]
    R = frame_float[:, :, 2]

    Y  =  0.299  * R + 0.587  * G + 0.114  * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5    * B + 128
    Cr =  0.5    * R - 0.4187 * G - 0.0813 * B + 128

    return Y, Cb, Cr


def chroma_subsample(Cb, Cr):
    """4:2:0 subsampling — reduce Cb and Cr by half in both dimensions."""
    Cb_sub = Cb[::2, ::2]
    Cr_sub = Cr[::2, ::2]
    return Cb_sub, Cr_sub


def chroma_upsample(Cb_sub, Cr_sub, original_shape):
    """Upsample Cb and Cr back to original size."""
    h, w = original_shape
    Cb_up = cv2.resize(Cb_sub, (w, h), interpolation=cv2.INTER_LINEAR)
    Cr_up = cv2.resize(Cr_sub, (w, h), interpolation=cv2.INTER_LINEAR)
    return Cb_up, Cr_up


def ycbcr_to_bgr(Y, Cb, Cr):
    """Convert YCbCr back to BGR."""
    R = Y + 1.402  * (Cr - 128)
    G = Y - 0.3441 * (Cb - 128) - 0.7141 * (Cr - 128)
    B = Y + 1.7720 * (Cb - 128)

    bgr = np.stack([B, G, R], axis=2)
    bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    return bgr


def preprocess_frame(frame_bgr):
    """Full preprocessing pipeline for one frame."""
    Y, Cb, Cr = bgr_to_ycbcr(frame_bgr)
    Cb_sub, Cr_sub = chroma_subsample(Cb, Cr)
    return Y, Cb_sub, Cr_sub


def reconstruct_frame(Y, Cb_sub, Cr_sub):
    """Reconstruct BGR frame from YCbCr components."""
    h, w = Y.shape
    Cb_up, Cr_up = chroma_upsample(Cb_sub, Cr_sub, (h, w))
    return ycbcr_to_bgr(Y, Cb_up, Cr_up)