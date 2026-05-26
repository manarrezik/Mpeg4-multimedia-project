# test_entropy.py
import cv2
import os
from preprocessing import preprocess_frame
from intra_coding import encode_iframe, decode_iframe
from inter_coding import encode_pframe, decode_pframe
from entropy_coding import encode_to_bin, decode_from_bin

# Simule 3 frames
frame1 = cv2.imread("frames/Rectangle 16.png")
frame2 = cv2.GaussianBlur(frame1, (5, 5), 0)
frame3 = cv2.GaussianBlur(frame1, (7, 7), 0)
frames = [frame1, frame2, frame3]

GOP_SIZE = 3
Quality  = 50
encoded_frames = []
ref_Y = None

for idx, frame in enumerate(frames):
    Y, Cb, Cr = preprocess_frame(frame)

    if idx % GOP_SIZE == 0:
        # I-frame
        enc = encode_iframe(Y, Cb, Cr, quality=Quality)
        Y_rec, _, _ = decode_iframe(enc)
        ref_Y = Y_rec
        print(f"Frame {idx} → I-frame")
    else:
        # P-frame
        enc = encode_pframe(Y, ref_Y, quality=Quality)
        Y_rec = decode_pframe(enc, ref_Y)
        ref_Y = Y_rec
        print(f"Frame {idx} → P-frame")

    encoded_frames.append(enc)

# Encode to .bin
encode_to_bin(encoded_frames, "output/compressed.bin")

# Decode from .bin
decoded_frames = decode_from_bin("output/compressed.bin")

# Stats
original_size = sum(f.nbytes for f in frames)
bin_size      = os.path.getsize("output/compressed.bin")

print(f"\n📊 Stats:")
print(f"   Frames encodées   : {len(decoded_frames)}")
print(f"   Original size     : {original_size:,} bytes")
print(f"   Compressed size   : {bin_size:,} bytes")
print(f"   Compression ratio : {original_size / bin_size:.2f}x")