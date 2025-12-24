import cv2
from colour_extractor import extract_dominant_colors
from wheel_generator import generate_color_wheel

IMAGE_PATH = "../images/img1.jpg"

def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("Image not found!")
        return

    dominant_colors = extract_dominant_colors(image, k=6)
    color_wheel = generate_color_wheel(dominant_colors)

    cv2.imshow("Original Image", image)
    cv2.imshow("Color Detection Wheel", color_wheel)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
