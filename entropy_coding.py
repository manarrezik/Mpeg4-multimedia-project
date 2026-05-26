# entropy_coding.py
import numpy as np
import pickle
import zlib
import os


def serialize_data(encoded_frames):
    """
    Serialize all encoded frames (I-frames and P-frames) into bytes using pickle.
    """
    return pickle.dumps(encoded_frames)


def compress_data(serialized_data):
    """
    Apply zlib lossless compression to serialized data.
    """
    return zlib.compress(serialized_data, level=9)  # level 9 = max compression


def decompress_data(compressed_data):
    """
    Decompress zlib compressed data.
    """
    return zlib.decompress(compressed_data)


def deserialize_data(serialized_data):
    """
    Deserialize bytes back into encoded frames.
    """
    return pickle.loads(serialized_data)


def write_bin_file(compressed_data, output_path):
    """
    Write compressed data to a .bin file with a header.
    Header format: magic bytes + version + data size
    """
    magic   = b'MPEG4SIM'   # 8 bytes identifier
    version = (1).to_bytes(2, byteorder='big')  # version 1
    size    = len(compressed_data).to_bytes(8, byteorder='big')  # data size

    with open(output_path, 'wb') as f:
        f.write(magic)
        f.write(version)
        f.write(size)
        f.write(compressed_data)

    print(f"✅ Written to {output_path} ({os.path.getsize(output_path)} bytes)")


def read_bin_file(input_path):
    """
    Read and validate a .bin file, return compressed data.
    """
    with open(input_path, 'rb') as f:
        magic   = f.read(8)
        version = int.from_bytes(f.read(2), byteorder='big')
        size    = int.from_bytes(f.read(8), byteorder='big')
        compressed_data = f.read()

    if magic != b'MPEG4SIM':
        raise ValueError(f"❌ Invalid file format: {magic}")

    if len(compressed_data) != size:
        raise ValueError(f"❌ Data size mismatch: expected {size}, got {len(compressed_data)}")

    print(f"✅ Read {input_path} | version={version} | size={size} bytes")
    return compressed_data


def encode_to_bin(encoded_frames, output_path):
    """
    Full pipeline: encoded frames → serialized → compressed → .bin file
    """
    print("🔄 Serializing data...")
    serialized = serialize_data(encoded_frames)
    print(f"   Serialized size : {len(serialized):,} bytes")

    print("🔄 Compressing data...")
    compressed = compress_data(serialized)
    print(f"   Compressed size : {len(compressed):,} bytes")
    print(f"   Compression ratio (entropy stage): {len(serialized)/len(compressed):.2f}x")

    print("🔄 Writing .bin file...")
    write_bin_file(compressed, output_path)


def decode_from_bin(input_path):
    """
    Full pipeline: .bin file → decompressed → deserialized → encoded frames
    """
    print("🔄 Reading .bin file...")
    compressed = read_bin_file(input_path)

    print("🔄 Decompressing data...")
    serialized = decompress_data(compressed)

    print("🔄 Deserializing data...")
    encoded_frames = deserialize_data(serialized)

    print(f"✅ Decoded {len(encoded_frames)} frames")
    return encoded_frames