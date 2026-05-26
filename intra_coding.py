# intra_coding.py
import numpy as np
from scipy.fft import dctn, idctn

# Matrice de quantification standard JPEG (luminance)
Q_MATRIX = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68,  109, 103, 77],
    [24, 35, 55, 64, 81,  104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)


def get_quantization_matrix(quality=50):
    """Scale Q matrix based on quality factor (1=low quality, 100=high quality)."""
    if quality <= 0:
        quality = 1
    if quality > 100:
        quality = 100

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    Q = np.floor((Q_MATRIX * scale + 50) / 100)
    Q = np.clip(Q, 1, 255)
    return Q


def apply_dct(block):
    """Apply 2D DCT to an 8x8 block."""
    return dctn(block, norm='ortho')


def apply_idct(block):
    """Apply 2D IDCT to an 8x8 block."""
    return idctn(block, norm='ortho')


def quantize(dct_block, Q):
    """Quantize DCT coefficients."""
    return np.round(dct_block / Q).astype(np.int32)


def dequantize(quant_block, Q):
    """Dequantize coefficients."""
    return (quant_block * Q).astype(np.float32)


def encode_channel(channel, quality=50):
    """
    Encode a single channel (Y, Cb, or Cr) using DCT + quantization.
    Returns quantized blocks as a 2D array of 8x8 blocks.
    """
    h, w = channel.shape
    Q = get_quantization_matrix(quality)

    # Pad to multiple of 8
    h_pad = (8 - h % 8) % 8
    w_pad = (8 - w % 8) % 8
    padded = np.pad(channel, ((0, h_pad), (0, w_pad)), mode='edge')

    ph, pw = padded.shape
    blocks_h = ph // 8
    blocks_w = pw // 8

    quantized_blocks = np.zeros((blocks_h, blocks_w, 8, 8), dtype=np.int32)

    for i in range(blocks_h):
        for j in range(blocks_w):
            block = padded[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32) - 128
            dct_block = apply_dct(block)
            quantized_blocks[i, j] = quantize(dct_block, Q)

    return quantized_blocks, (h, w), quality


def decode_channel(quantized_blocks, original_shape, quality=50):
    """
    Decode a channel from quantized DCT blocks back to pixel values.
    """
    Q = get_quantization_matrix(quality)
    blocks_h, blocks_w = quantized_blocks.shape[:2]
    ph, pw = blocks_h * 8, blocks_w * 8

    reconstructed = np.zeros((ph, pw), dtype=np.float32)

    for i in range(blocks_h):
        for j in range(blocks_w):
            dct_block = dequantize(quantized_blocks[i, j], Q)
            block = apply_idct(dct_block) + 128
            reconstructed[i*8:(i+1)*8, j*8:(j+1)*8] = block

    # Crop back to original size
    h, w = original_shape
    reconstructed = reconstructed[:h, :w]
    return np.clip(reconstructed, 0, 255).astype(np.float32)


def encode_iframe(Y, Cb, Cr, quality=50):
    """Encode a full I-frame (all 3 channels)."""
    Y_blocks,  Y_shape,  _ = encode_channel(Y,  quality)
    Cb_blocks, Cb_shape, _ = encode_channel(Cb, quality)
    Cr_blocks, Cr_shape, _ = encode_channel(Cr, quality)
    return {
        'type'    : 'I',
        'Y'       : Y_blocks,
        'Cb'      : Cb_blocks,
        'Cr'      : Cr_blocks,
        'Y_shape' : Y_shape,
        'Cb_shape': Cb_shape,
        'Cr_shape': Cr_shape,
        'quality' : quality
    }


def decode_iframe(iframe_data):
    """Decode a full I-frame."""
    quality = iframe_data['quality']
    Y  = decode_channel(iframe_data['Y'],  iframe_data['Y_shape'],  quality)
    Cb = decode_channel(iframe_data['Cb'], iframe_data['Cb_shape'], quality)
    Cr = decode_channel(iframe_data['Cr'], iframe_data['Cr_shape'], quality)
    return Y, Cb, Cr