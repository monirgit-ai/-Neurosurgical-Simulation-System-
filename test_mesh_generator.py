from image_loader import load_image
from image_preprocessing import preprocess_image, auto_segment
from mesh_generator import create_mesh_from_mask, visualize_mesh

image, _ = load_image("assets/sample_image.dcm")
preprocessed = preprocess_image(image)
mask = auto_segment(preprocessed)

mesh = create_mesh_from_mask(mask)
visualize_mesh(mesh)
