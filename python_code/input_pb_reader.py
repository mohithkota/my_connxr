import numpy as np

with open("/home/mohithkota/connx/connx_ajit/test/mnist/test_data_set_0/input_0.pb", "rb") as f:
    data = f.read()

print("Total bytes in file:", len(data))

# Assume each image is 784 bytes (28x28)
num_full_images = len(data) // 784
valid_data = data[:num_full_images * 784]

images = np.frombuffer(valid_data, dtype=np.uint8).reshape((-1, 28, 28))
print("Extracted", images.shape[0], "images")
print("Image shape:", images.shape[1:])

# Prepare one image for ONNX input
img = images[0].astype(np.float32) / 255.0
input_tensor = img.reshape(1, 1, 28, 28)  # Shape: (N,C,H,W)
print("Input tensor shape:", input_tensor.shape)
