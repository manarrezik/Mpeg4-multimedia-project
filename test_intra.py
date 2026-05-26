# test_intra.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocess_frame
from intra_coding import encode_iframe, decode_iframe

frame = cv2.imread("frames/Rectangle 16.png")
Y, Cb, Cr = preprocess_frame(frame)

# Encode
iframe = encode_iframe(Y, Cb, Cr, quality=50)

# Decode
Y_rec, Cb_rec, Cr_rec = decode_iframe(iframe)

# Visualise un bloc 8x8
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(Y[:8, :8], cmap='gray')
axes[0].set_title("Bloc original (8x8)")
axes[1].imshow(iframe['Y'][0, 0], cmap='gray')
axes[1].set_title("Coefficients DCT quantifiés")
axes[2].imshow(Y_rec[:8, :8], cmap='gray')
axes[2].set_title("Bloc reconstruit")
plt.tight_layout()
plt.savefig("output/test_intra.png")
plt.show()

print(f"Y original shape  : {Y.shape}")
print(f"Y blocks shape    : {iframe['Y'].shape}")
print(f"Y reconstructed   : {Y_rec.shape}")
print(f"Max difference    : {np.max(np.abs(Y - Y_rec)):.2f}")