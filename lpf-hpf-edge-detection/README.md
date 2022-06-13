# Image-Filtering
Image filtering using high-pass and low-pass filter

Original Image:

![Original Picture](sample01.jpg)

Low Pass Filter:


![Original Picture](low-pass.jpg)

High Pass Filter:


![Original Picture](high-pass.jpg)


Edge_detection:

def main():
    print("This might take some time please be patient ...")
    # Current Directory
    current_dir = os.getcwd()

    # Read Image from file_path

    file_path = ""  # <---------------insert image path here
    img = cv2.imread(file_path, 0)

    # Apply Canny Edge Detector
    final_img = Canny(img)

    # Show Final image
    cv2.imshow("5 final image", final_img)

    cv2.waitKey(0)

