import numpy as np
import cv2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def motion_blur_kernel(size=10, angle=45):
    kernel = np.zeros((size, size))
    center = size // 2
    angle_rad = np.deg2rad(angle)
    x = int(center + center * np.cos(angle_rad))
    y = int(center + center * np.sin(angle_rad))
    cv2.line(kernel, (center, center), (x, y), 1, thickness=1)
    kernel /= np.sum(kernel)
    return kernel

def apply_filter(img_channel, kernel):
    return convolve2d(img_channel, kernel, mode='same', boundary='wrap')

def reverse_filtering(observed, filter_func, iterations=20):
    current = observed.copy()
    for _ in range(iterations):
        filtered_current = filter_func(current)
        numerator = np.fft.fft2(observed) * np.fft.fft2(current)
        denominator = np.fft.fft2(filtered_current) + 1e-8
        current = np.real(np.fft.ifft2(numerator / denominator))
    return current

def normalize_image(img):
    return np.clip(img, 0, 1)

def main():
    # Load and normalize image
    img = cv2.imread("building_roof.jpg")
    if img is None:
        raise FileNotFoundError("Image 'building_roof.jpg' not found.")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]

    # Create motion blur filter
    kernel = motion_blur_kernel(10, 45)
    filter_func = lambda x: apply_filter(x, kernel)

    # Apply filter
    Ry, Gy, By = filter_func(R), filter_func(G), filter_func(B)

    # Reverse filtering
    Xrec_R = reverse_filtering(Ry, filter_func)
    Xrec_G = reverse_filtering(Gy, filter_func)
    Xrec_B = reverse_filtering(By, filter_func)

    # Stack channels and normalize
    filtered_img = np.stack([Ry, Gy, By], axis=-1)
    restored_img = np.stack([Xrec_R, Xrec_G, Xrec_B], axis=-1)
    filtered_img = normalize_image(filtered_img)
    restored_img = normalize_image(restored_img)

    # Show results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(filtered_img)
    plt.title("Filtered Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(restored_img)
    plt.title("Restored Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
