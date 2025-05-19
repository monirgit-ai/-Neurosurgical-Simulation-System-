from image_loader import load_image
import matplotlib.pyplot as plt

def test_image(file_path):
    try:
        image, metadata = load_image(file_path)
        print("✅ Image loaded successfully.")
        print("Metadata:")
        for k, v in metadata.items():
            print(f"  {k}: {v}")

        plt.imshow(image, cmap='gray')
        plt.title("Loaded Image Preview")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print("❌ Error:", e)

# Example test
test_image("assets/sample_image.dcm")  # Replace with actual path
