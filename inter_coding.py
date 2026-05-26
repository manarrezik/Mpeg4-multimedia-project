# inter_coding.py
import numpy as np
from intra_coding import encode_channel, decode_channel, apply_dct, apply_idct, quantize, dequantize, get_quantization_matrix


def block_matching(current_block, ref_frame, ref_y, ref_x, search_window=8):
    """
    Find the best matching block in the reference frame.
    Returns the motion vector (dy, dx) and the best matching block.
    """
    h, w = ref_frame.shape
    bh, bw = current_block.shape  # 16x16

    best_sad = float('inf')
    best_dy, best_dx = 0, 0

    for dy in range(-search_window, search_window + 1):
        for dx in range(-search_window, search_window + 1):
            ny = ref_y + dy
            nx = ref_x + dx

            # Boundary check
            if ny < 0 or nx < 0 or ny + bh > h or nx + bw > w:
                continue

            ref_block = ref_frame[ny:ny+bh, nx:nx+bw]
            sad = np.sum(np.abs(current_block.astype(np.float32) - ref_block.astype(np.float32)))

            if sad < best_sad:
                best_sad = sad
                best_dy = dy
                best_dx = dx

    best_ref_block = ref_frame[
        ref_y + best_dy : ref_y + best_dy + bh,
        ref_x + best_dx : ref_x + best_dx + bw
    ]
    return best_dy, best_dx, best_ref_block


def encode_residual(residual, quality=50):
    """Apply DCT + quantization to a residual block (16x16 split into 4 8x8 blocks)."""
    Q = get_quantization_matrix(quality)
    quant_blocks = np.zeros((2, 2, 8, 8), dtype=np.int32)

    for i in range(2):
        for j in range(2):
            block = residual[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32)
            dct_block = apply_dct(block)
            quant_blocks[i, j] = quantize(dct_block, Q)

    return quant_blocks


def decode_residual(quant_blocks, quality=50):
    """Decode residual from quantized DCT blocks."""
    Q = get_quantization_matrix(quality)
    residual = np.zeros((16, 16), dtype=np.float32)

    for i in range(2):
        for j in range(2):
            dct_block = dequantize(quant_blocks[i, j], Q)
            block = apply_idct(dct_block)
            residual[i*8:(i+1)*8, j*8:(j+1)*8] = block

    return residual


def encode_pframe(current_Y, ref_Y, quality=50, search_window=8):
    """
    Encode a P-frame using motion estimation on the Y channel.
    Returns motion vectors and encoded residuals.
    """
    h, w = current_Y.shape

    # Pad to multiple of 16
    h_pad = (16 - h % 16) % 16
    w_pad = (16 - w % 16) % 16
    curr_padded = np.pad(current_Y, ((0, h_pad), (0, w_pad)), mode='edge')
    ref_padded  = np.pad(ref_Y,     ((0, h_pad), (0, w_pad)), mode='edge')

    ph, pw = curr_padded.shape
    blocks_h = ph // 16
    blocks_w = pw // 16

    motion_vectors = np.zeros((blocks_h, blocks_w, 2), dtype=np.int32)
    residual_blocks = []

    for i in range(blocks_h):
        row = []
        for j in range(blocks_w):
            curr_block = curr_padded[i*16:(i+1)*16, j*16:(j+1)*16]

            dy, dx, ref_block = block_matching(
                curr_block, ref_padded,
                ref_y=i*16, ref_x=j*16,
                search_window=search_window
            )

            motion_vectors[i, j] = [dy, dx]

            residual = curr_block.astype(np.float32) - ref_block.astype(np.float32)
            quant_residual = encode_residual(residual, quality)
            row.append(quant_residual)

        residual_blocks.append(row)

    return {
        'type'           : 'P',
        'motion_vectors' : motion_vectors,
        'residuals'      : residual_blocks,
        'original_shape' : (h, w),
        'quality'        : quality
    }


def decode_pframe(pframe_data, ref_Y):
    """Reconstruct P-frame from motion vectors + residuals + reference frame."""
    h, w = pframe_data['original_shape']
    quality = pframe_data['quality']
    motion_vectors = pframe_data['motion_vectors']
    residual_blocks = pframe_data['residuals']

    h_pad = (16 - h % 16) % 16
    w_pad = (16 - w % 16) % 16
    ref_padded = np.pad(ref_Y, ((0, h_pad), (0, w_pad)), mode='edge')

    ph, pw = ref_padded.shape
    reconstructed = np.zeros((ph, pw), dtype=np.float32)

    blocks_h = ph // 16
    blocks_w = pw // 16

    for i in range(blocks_h):
        for j in range(blocks_w):
            dy, dx = motion_vectors[i, j]
            ref_y = i * 16 + dy
            ref_x = j * 16 + dx

            ref_y = np.clip(ref_y, 0, ph - 16)
            ref_x = np.clip(ref_x, 0, pw - 16)

            ref_block = ref_padded[ref_y:ref_y+16, ref_x:ref_x+16].astype(np.float32)
            residual = decode_residual(residual_blocks[i][j], quality)

            reconstructed[i*16:(i+1)*16, j*16:(j+1)*16] = ref_block + residual

    reconstructed = reconstructed[:h, :w]
    return np.clip(reconstructed, 0, 255).astype(np.float32)


def get_frame_type(frame_index, gop_size):
    """Returns 'I' or 'P' based on frame index and GOP size."""
    if frame_index % gop_size == 0:
        return 'I'
    return 'P'