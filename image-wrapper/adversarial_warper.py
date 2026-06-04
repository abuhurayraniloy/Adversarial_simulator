import numpy as np
from PIL import Image

def adversarial_stress_test(image_path, output_path, shear_factor, noise_severity):
    img = Image.open(image_path)
    np_img = np.array(img)
    h, w, c = np_img.shape
    y_indices, x_indices = np.indices((h, w))
    
    cx = w / 2.0
    cy = h / 2.0

    x_centered = x_indices - cx
    y_centered = y_indices - cy

    target_coords = np.vstack([x_centered.ravel(), y_centered.ravel()])
    
    distance = np.sqrt(target_coords[0]**2 + target_coords[1]**2)
    normalized_distance = distance / np.max(distance)
   
    M_forward = np.array([[1, shear_factor],
                          [0, 1]])
    
    M_inverse = np.linalg.inv(M_forward)
    source_coords = M_inverse @ target_coords

    noise_x = np.random.normal(0, noise_severity, size=source_coords.shape[1]) * normalized_distance
    noise_y = np.random.normal(0, noise_severity, size=source_coords.shape[1]) * normalized_distance

    source_coords[0] += noise_x
    source_coords[1] += noise_y

    x_source = source_coords[0] + cx
    y_source = source_coords[1] + cy

    x_source = np.clip(np.round(x_source).astype(int), 0, w - 1)
    y_source = np.clip(np.round(y_source).astype(int), 0, h - 1)

    stressed_img_arr = np_img[y_source, x_source].reshape((h, w, c))

    stressed_img = Image.fromarray(stressed_img_arr.astype('uint8'))
    stressed_img.save(output_path)
    print(f"Successfully generated adversarial sample: {output_path}")

if __name__ == "__main__":
    try:
        adversarial_stress_test(
            image_path="img.jpg", 
            output_path="output_image.jpg", 
            shear_factor=0.7, 
            noise_severity=8.0
        )
    except FileNotFoundError:
        print("Input image not found. Please upload the photo in the current directory.")