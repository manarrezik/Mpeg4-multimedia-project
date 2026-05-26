# test_preprocessing.py
import cv2
import matplotlib.pyplot as plt
from preprocessing import preprocess_frame, reconstruct_frame

# Charge l'image
frame = cv2.imread("frames/Rectangle 16.png")

if frame is None:
    print("❌ Image non trouvée !")
else:
    print(f"✅ Image chargée : {frame.shape}")

# Applique le preprocessing
Y, Cb_sub, Cr_sub = preprocess_frame(frame)

# Reconstruit
reconstructed = reconstruct_frame(Y, Cb_sub, Cr_sub)

# Affiche les résultats
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Frame Originale")

axes[0, 1].imshow(Y, cmap='gray')
axes[0, 1].set_title("Canal Y (Luminance)")

axes[0, 2].imshow(Cb_sub, cmap='gray')
axes[0, 2].set_title("Canal Cb (sous-échantillonné)")

axes[1, 0].imshow(Cr_sub, cmap='gray')
axes[1, 0].set_title("Canal Cr (sous-échantillonné)")

axes[1, 1].imshow(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("Frame Reconstruite")

axes[1, 2].imshow(cv2.absdiff(frame, reconstructed))
axes[1, 2].set_title("Différence (erreur)")

plt.tight_layout()
plt.savefig("output/test_preprocessing.png")
plt.show()

print(f"Y shape     : {Y.shape}")
print(f"Cb subsampled : {Cb_sub.shape}")
print(f"Cr subsampled : {Cr_sub.shape}")