# test_inter.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocess_frame
from intra_coding import encode_iframe, decode_iframe
from inter_coding import encode_pframe, decode_pframe

# On simule 2 frames légèrement différentes
frame1 = cv2.imread("frames/Rectangle 16.png")
frame2 = frame1.copy()
frame2 = cv2.GaussianBlur(frame2, (5, 5), 0)  # légère différence

# Preprocess
Y1, Cb1, Cr1 = preprocess_frame(frame1)
Y2, Cb2, Cr2 = preprocess_frame(frame2)

# Frame 1 = I-frame (référence)
iframe = encode_iframe(Y1, Cb1, Cr1, quality=50)
Y1_rec, _, _ = decode_iframe(iframe)

# Frame 2 = P-frame
pframe = encode_pframe(Y2, Y1_rec, quality=50, search_window=8)
Y2_rec = decode_pframe(pframe, Y1_rec)

# Visualisation
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

axes[0].imshow(Y1, cmap='gray')
axes[0].set_title("Frame 1 (référence Y)")

axes[1].imshow(Y2, cmap='gray')
axes[1].set_title("Frame 2 originale Y")

axes[2].imshow(Y2_rec, cmap='gray')
axes[2].set_title("Frame 2 reconstruite Y")

# Motion vectors
mv = pframe['motion_vectors']
ax = axes[3]
ax.imshow(Y2, cmap='gray')
step = 1
for i in range(0, mv.shape[0], step):
    for j in range(0, mv.shape[1], step):
        dy, dx = mv[i, j]
        ax.annotate("", xy=(j*16+8+dx, i*16+8+dy),
                    xytext=(j*16+8, i*16+8),
                    arrowprops=dict(arrowstyle="->", color='red', lw=1))
axes[3].set_title("Motion Vectors")

plt.tight_layout()
plt.savefig("output/test_inter.png")
plt.show()

print(f"Motion vectors shape : {mv.shape}")
print(f"Y2 reconstructed shape : {Y2_rec.shape}")
print(f"Max difference : {np.max(np.abs(Y2 - Y2_rec)):.2f}")