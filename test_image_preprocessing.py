from image_loader import load_image
from image_preprocessing import preprocess_image, auto_segment, manual_segment
import matplotlib.pyplot as plt

# Load image
image, meta = load_image("assets/sample_image.dcm")

# Preprocess
preprocessed = preprocess_image(image)

# Show original vs enhanced
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(preprocessed, cmap='gray')
plt.title("Preprocessed")
plt.show()

# Auto segmentation
auto_mask = auto_segment(preprocessed)
plt.imshow(auto_mask, cmap='gray')
plt.title("Auto Segmentation")
plt.show()

# Optional: Manual segmentation
manual_mask = manual_segment(preprocessed)
plt.imshow(manual_mask, cmap='gray')
plt.title("Manual Segmentation")
plt.show()
