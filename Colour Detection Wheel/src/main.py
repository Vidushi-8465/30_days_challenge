import cv2 # imports open cv for image reading and display
from colour_extractor import extract_dominant_colors  # imports function that will extract the dominant colours from the image
from wheel_generator import generate_color_wheel      # imports function that will generate the colour wheel for the selected image

IMAGE_PATH = "../images/img1.jpg"   # The path to the image

def main():
    image = cv2.imread(IMAGE_PATH)   # To read the image from disk in BGR format which is opencv's default

    if image is None:     # if no image is found or the path is incoreect 
        print("Image not found!")
        return

    dominant_colours = extract_dominant_colours(image, k=6)   # It calls the dominant colour function which finds the "K" number of colours
    colour_wheel = generate_colour_wheel(dominant_colours)   # it calls the generate colour wheel function to generate the wheel of the extracted colours

    cv2.imshow("Original Image", image)    # To show the original colour in the output
    cv2.imshow("Colour Detection Wheel", colour_wheel)  # To show the colour detection wheel in the output along with original image

    cv2.waitKey(0)  # waits until any key is pressend
    cv2.destroyAllWindows()  # closses any other opencv windows
 
if __name__ == "__main__":
    main()
