# evaluation.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import os


# ─── Quality Metrics ────────────────────────────────────────────────────────

def compute_psnr(original, reconstructed):
    """Compute Peak Signal-to-Noise Ratio between two frames."""
    original      = original.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_compression_ratio(original_frames, bin_path):
    """Compute overall compression ratio."""
    original_size  = sum(f.nbytes for f in original_frames)
    compressed_size = os.path.getsize(bin_path)
    return original_size / compressed_size, original_size, compressed_size


def print_metrics(frame_types, psnr_values, original_frames, bin_path):
    """Print a summary of all metrics."""
    ratio, orig_size, comp_size = compute_compression_ratio(original_frames, bin_path)

    print("\n" + "="*50)
    print("         EVALUATION SUMMARY")
    print("="*50)
    print(f"  Total frames     : {len(frame_types)}")
    print(f"  I-frames         : {frame_types.count('I')}")
    print(f"  P-frames         : {frame_types.count('P')}")
    print(f"  Original size    : {orig_size:,} bytes")
    print(f"  Compressed size  : {comp_size:,} bytes")
    print(f"  Compression ratio: {ratio:.2f}x")
    print(f"  Avg PSNR         : {np.mean(psnr_values):.2f} dB")
    print("="*50)
    for i, (ft, psnr) in enumerate(zip(frame_types, psnr_values)):
        print(f"  Frame {i:2d} [{ft}]  PSNR: {psnr:.2f} dB")
    print("="*50 + "\n")


# ─── Pipeline Visualisation ─────────────────────────────────────────────────

def visualize_pipeline(original_frames, reconstructed_frames,
                       encoded_frames, frame_types,
                       psnr_values, bin_path):
    """
    Produce a single figure showing every stage of the pipeline.
    """
    from preprocessing import preprocess_frame
    from intra_coding  import encode_iframe

    fig = plt.figure(figsize=(22, 24))
    fig.suptitle("MPEG-4 Encoder Pipeline — Full Visualisation", fontsize=16, fontweight='bold')

    # ── 1. Original frames ──────────────────────────────────────────────────
    n_frames = min(len(original_frames), 4)
    for i in range(n_frames):
        ax = fig.add_subplot(6, 4, i + 1)
        ax.imshow(cv2.cvtColor(original_frames[i], cv2.COLOR_BGR2RGB))
        ax.set_title(f"Frame {i} [{frame_types[i]}]", fontsize=9)
        ax.axis('off')

    # ── 2. Color space channels ─────────────────────────────────────────────
    Y, Cb, Cr = preprocess_frame(original_frames[0])
    for idx, (channel, name) in enumerate([(Y, 'Y'), (Cb, 'Cb'), (Cr, 'Cr')]):
        ax = fig.add_subplot(6, 4, 5 + idx)
        ax.imshow(channel, cmap='gray')
        ax.set_title(f"Color Space: {name}", fontsize=9)
        ax.axis('off')

    ax_info = fig.add_subplot(6, 4, 8)
    ax_info.text(0.5, 0.5,
                 f"YCbCr 4:2:0\nY:  {Y.shape}\nCb: {Cb.shape}\nCr: {Cr.shape}",
                 ha='center', va='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax_info.axis('off')
    ax_info.set_title("Subsampling Info", fontsize=9)

    # ── 3. DCT & Quantisation ────────────────────────────────────────────────
    # Find a non-uniform block
    bi, bj = 0, 0
    for ii in range(encoded_frames[0]['Y'].shape[0]):
        for jj in range(encoded_frames[0]['Y'].shape[1]):
            if Y[ii*8:(ii+1)*8, jj*8:(jj+1)*8].std() > 10:
                bi, bj = ii, jj
                break
        else:
            continue
        break

    raw_block  = Y[bi*8:(bi+1)*8, bj*8:(bj+1)*8]
    dct_block  = encoded_frames[0]['Y'][bi, bj]
    rec_block  = reconstructed_frames[0][bi*8:(bi+1)*8, bj*8:(bj+1)*8] if reconstructed_frames[0] is not None else raw_block

    for idx, (data, title) in enumerate([
        (raw_block,  "Raw 8×8 block"),
        (dct_block,  "DCT quantised"),
        (rec_block,  "Reconstructed block")
    ]):
        ax = fig.add_subplot(6, 4, 9 + idx)
        im = ax.imshow(data, cmap='RdBu_r')
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis('off')

    # ── 4. Motion vectors ────────────────────────────────────────────────────
    p_idx = next((i for i, t in enumerate(frame_types) if t == 'P'), None)
    ax_mv = fig.add_subplot(6, 4, 13)

    if p_idx is not None and reconstructed_frames[p_idx - 1] is not None:
        ax_mv.imshow(reconstructed_frames[p_idx - 1], cmap='gray')
        mv = encoded_frames[p_idx]['motion_vectors']
        for i in range(0, mv.shape[0], 2):
            for j in range(0, mv.shape[1], 2):
                dy, dx = mv[i, j]
                ax_mv.annotate("", xy=(j*16+8+dx, i*16+8+dy),
                               xytext=(j*16+8, i*16+8),
                               arrowprops=dict(arrowstyle="->", color='red', lw=0.8))
        ax_mv.set_title(f"Motion Vectors (Frame {p_idx})", fontsize=9)
    else:
        ax_mv.text(0.5, 0.5, "No P-frame", ha='center', va='center')
        ax_mv.set_title("Motion Vectors", fontsize=9)
    ax_mv.axis('off')

    # ── 5. Residuals & Reconstruction ────────────────────────────────────────
    if p_idx is not None:
        mv   = encoded_frames[p_idx]['motion_vectors']
        dy0, dx0 = mv[0, 0]
        h_ref, w_ref = reconstructed_frames[p_idx - 1].shape
        ref_y = min(max(0, dy0), h_ref - 16)
        ref_x = min(max(0, dx0), w_ref - 16)
        ref_block_16 = reconstructed_frames[p_idx - 1][ref_y:ref_y+16, ref_x:ref_x+16]

        from inter_coding import decode_residual
        res_decoded = decode_residual(encoded_frames[p_idx]['residuals'][0][0],
                                      encoded_frames[p_idx]['quality'])

        ax_res = fig.add_subplot(6, 4, 14)
        ax_res.imshow(res_decoded, cmap='RdBu_r')
        ax_res.set_title("Residual (P-frame block)", fontsize=9)
        ax_res.axis('off')

        ax_ref = fig.add_subplot(6, 4, 15)
        ax_ref.imshow(ref_block_16, cmap='gray')
        ax_ref.set_title("Ref block (motion comp.)", fontsize=9)
        ax_ref.axis('off')

    # ── 6. PSNR chart ────────────────────────────────────────────────────────
    ax_psnr = fig.add_subplot(6, 1, 6)
    colors = ['steelblue' if t == 'I' else 'tomato' for t in frame_types]
    bars   = ax_psnr.bar(range(len(psnr_values)), psnr_values, color=colors)
    ax_psnr.set_xlabel("Frame index")
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.set_title("PSNR per Frame")
    ax_psnr.axhline(y=np.mean(psnr_values), color='green', linestyle='--',
                    label=f"Mean PSNR: {np.mean(psnr_values):.2f} dB")
    ax_psnr.legend()
    i_patch = mpatches.Patch(color='steelblue', label='I-frame')
    p_patch = mpatches.Patch(color='tomato',    label='P-frame')
    ax_psnr.legend(handles=[i_patch, p_patch,
                             mpatches.Patch(color='green', label=f"Mean: {np.mean(psnr_values):.2f} dB")])

    plt.tight_layout()
    plt.savefig("output/pipeline_visualisation.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: output/pipeline_visualisation.png")