import sys
import struct
from pathlib import Path

def is_probably_protobuf(data):
    # Heuristics:
    # - Protobuf usually starts with small integers as field keys
    # - Many fields have wire type 2 (length-delimited)
    suspicious = 0
    i = 0
    while i < len(data) and i < 512:
        tag = data[i]
        wire_type = tag & 0x07
        field_number = tag >> 3
        i += 1

        if wire_type == 2 and i < len(data):
            length = data[i]
            i += 1
            i += length
            suspicious += 1
        elif wire_type == 0:
            while i < len(data) and (data[i] & 0x80):
                i += 1
            i += 1
        else:
            i += 1

        if suspicious >= 5:
            return True
    return False

def has_human_strings(data):
    # Check for readable text
    text = data[:2048].decode("utf-8", errors="ignore")
    keywords = ["onnx", "tensorflow", "input", "graph", "node", "op_type"]
    return any(k in text.lower() for k in keywords)

def detect_file_type(path):
    with open(path, "rb") as f:
        data = f.read()

    size = len(data)
    print(f"File: {path} ({size} bytes)")

    # Simple checks
    if data[:4] == b'TFL3':
        return "TensorFlow Lite model (FlatBuffer)"
    if data[:2] == b'PK':
        return "Zip file (maybe SavedModel)"
    if b"onnx" in data[:1024]:
        return "ONNX model (likely)"
    if b"input" in data and b"node" in data:
        return "TensorFlow GraphDef (likely)"
    if is_probably_protobuf(data):
        if has_human_strings(data):
            return "Protobuf-encoded ONNX or TF model"
        else:
            return "Unknown protobuf binary (no strings found)"
    if size % 784 == 0:
        return "Raw MNIST image blob (28x28 grayscale)"
    if size % 4 == 0:
        return "Likely raw float32 binary blob"

    return "Unknown / Non-protobuf raw binary"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_pb_type.py <file.pb>")
        sys.exit(1)

    result = detect_file_type(sys.argv[1])
    print("Detected type:", result)
