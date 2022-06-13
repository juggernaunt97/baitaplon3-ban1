import os
import numpy as np
import cv2 as cv2
import math
from sympy import symbols
from scipy import ndimage


# Blur Functions


def gaussianMask(mask_size=3, Sigma=1):
# make a gaussian mask 3*3 by making 2 dense matrices and apply the gaussian function to them 
    x, y = symbols("x y")
    my_Mask = []
    X = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
    Y = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
    for i in range(len(X)):

        for j in range(len(X[0])):
            x_val = X[i][j]
            y_val = Y[i][j]
            exponent = -1 * (((x * x) + (y * y)) / (2 * 1 * 1))
            exponent = exponent.subs(x, x_val)
            exponent = exponent.subs(y, y_val)
            result_e = math.exp(exponent) # e power x^2+y^2 / 2*sigma^2
            expr = 1 / 2 * math.pi * 1 * 1
            result = round(expr * result_e)  # 1/2 * pi *sigma^2  * e power x^2+y^2 / 2*sigma^2
            my_Mask.append(result)
    my_Mask = np.array(my_Mask)
    my_Mask = np.reshape(my_Mask, (3, 3))
    return my_Mask


def convolute(img, mask):
    # my_filtered_img = np.convolve(image, mask)
    img_len = len(img)
    row_list = []
    M_len = len(mask)
    # Computing The gradient for each pixel
    for i in range(0, img_len):
        col_list = []
        width_len = len(img[i])
        for j in range(0, width_len):

            # We can't compute the gradient for the edges
            if i == 0 or j == 0 or i == img_len - 1 or j == width_len - 1:
                col_list.append(img[i][j])
            else:
                sum = 0
                for a in range(0, M_len):
                    for b in range(0, M_len):
                        sum = +mask[a][b] * img[i - 1 + a][j - 1 + b]
                col_list.append(sum / (M_len * M_len))
        row_list.append(col_list)
    new_img = np.array(row_list, dtype=np.uint8)
    return new_img


# Get gradient and direction of edges


def applyEdgeDetector(img, Mx, My):

    M_len = len(Mx)
    img_len = len(img)

    row_list = []
    direction_list = []
    # Computing The gradient for each pixel
    for i in range(0, img_len):
        col_list = []
        direction_col_list = []
        width_len = len(img[i])
        for j in range(0, width_len):

            # We can't compute the gradient for the edges
            if i == 0 or j == 0 or i == img_len - 1 or j == width_len - 1:
                col_list.append(img[i][j])
                direction_col_list.append(0)
            else:
                magX = 0
                magY = 0

                # Compute Gx and Gy
                for a in range(0, M_len):
                    for b in range(0, M_len):
                        magX += Mx[a][b] * img[i - 1 + a][j - 1 + b]
                        magY += My[a][b] * img[i - 1 + a][j - 1 + b]

                # The resulting gradient
                mag = math.sqrt((magX ** 2) + (magY ** 2))
                col_list.append(mag)

                # The Direction of edge
                if magX == 0:
                    direction = 90
                else:
                    direction = math.degrees(math.atan(magY / magX))
                direction_col_list.append(direction)

        row_list.append(col_list)
        direction_list.append(direction_col_list)
    # convert 2D list to 2D unsigned 8-bit array
    new_img = np.array(row_list, dtype=np.uint8)
    return new_img, direction_list


def firstDerivativeEdgeDetector(image):
    Mx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]] # the matrix centered about x-axis
    My = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]] #the matrix centered about y-axis
    new_img, direction = applyEdgeDetector(image, Mx, My)
    return new_img, direction


def secondDerivativeEdgeDetector(image):
    Mx = [[1, -2, 1], [1, -2, 1], [1, -2, 1]]
    My = [[1, 1, 1], [-2, -2, -2], [1, 1, 1]]
    new_img, direction = applyEdgeDetector(image, Mx, My)
    return new_img, direction


def sobel(img):
    Mx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    My = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    new_img, direction = applyEdgeDetector(img, Mx, My)
    return new_img, direction


def prewitt(img):
    Mx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    My = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
    new_img, direction = applyEdgeDetector(img, Mx, My)
    return new_img, direction


# Thin Edges


def NonMaxima_suppression(img, direction):

    # apply NonMaxima supression
    img_len = len(img)
    row_list = []
    for i in range(0, img_len):
        col_list = []
        width_len = len(img[i])
        for j in range(0, width_len):

            # We can't compute the gradient for the edges
            if i == 0 or j == 0 or i == img_len - 1 or j == width_len - 1:
                col_list.append(0)

            else:
                following_pixel = 255
                pre_pixel = 255
                theta = direction[i][j]
                # horizontal direction
                if theta <= 22.5 and theta >= -22.5:
                    following_pixel = img[i][j + 1]
                    pre_pixel = img[i][j - 1]
                # vertical direction
                elif theta <= -67.5 or theta >= 67.5:
                    following_pixel = img[i + 1][j]
                    pre_pixel = img[i - 1][j]
                # upper left to bottom right
                elif theta < -22.5 and theta > -67.5:
                    following_pixel = img[i + 1][j - 1]
                    pre_pixel = img[i - 1][j + 1]
                # bottom left to upper right
                elif theta < 67.5 and theta > 22.5:
                    following_pixel = img[i + 1][j + 1]
                    pre_pixel = img[i - 1][j - 1]

                if img[i][j] > following_pixel and pre_pixel > following_pixel:
                    col_list.append(img[i][j])
                else:
                    col_list.append(0)
        row_list.append(col_list)
    new_img = np.array(row_list, dtype=np.uint8)
    return new_img


# Double_threshold


def double_threshold(img):
  # put upper and lower boundries to chech if edge is weak or strong based on its position between the boundries
    img_len = len(img)
    max_intensity = img.max()
    highThreshold = max_intensity * 0.04
    lowThreshold = highThreshold * 0.03
    row_list = []
    for i in range(0, img_len):
        col_list = []
        width_len = len(img[i])
        for j in range(0, width_len):
            if img[i][j] > highThreshold:
                new_pixel = 255
            elif img[i][j] < lowThreshold:
                new_pixel = 0
            else:
                new_pixel = 60
            col_list.append(new_pixel)
        row_list.append(col_list)

    new_img = np.array(row_list, dtype=np.uint8)
    return new_img


# Edge_Linking


def Edge_Linking(img):
    # check the neighbors of each weak edge pixel to check if there is a strong edge 
    # if so set the value of the pixel to 255 else make it black =0 
    # that will show alll the edges and will connect them 
    img_len = len(img)
    row_list = []

    for i in range(img_len):

        col_list = []
        width_len = len(img[i])

        for j in range(len(img[i])):

            new_pixel = 0

            if img[i][j] == 60:

                if i == 0 or j == 0 or i == img_len - 1 or j == width_len - 1:
                    pass

                elif (
                    (img[i - 1][j - 1] == 255)
                    or (img[i - 1][j] == 255)
                    or (img[i - 1][j + 1] == 255)
                    or (img[i][j - 1] == 255)
                    or (img[i][j + 1] == 255)
                    or (img[i + 1][j - 1] == 255)
                    or (img[i + 1][j] == 255)
                    or (img[i + 1][j + 1] == 255)
                ):
                    new_pixel = 255

            elif img[i][j] == 255:

                new_pixel = 255

            col_list.append(new_pixel)

        row_list.append(col_list)

    new_img = np.array(row_list, dtype=np.uint8)

    return new_img


# Canny Edge Detecor


def Canny(img):

    # Generate Mask
    mask = gaussianMask()

    # Applying Filter
    bluredImg = convolute(img, mask)

    # Apply  any of the Edge Detectors

    ImgEdge, direction = sobel(bluredImg)  # Sobel is the default Edge detector
    # ImgEdge, direction = firstDerivativeEdgeDetector(bluredImg) #Uncomment to use first Derivative Edge Detector
    # ImgEdge, direction = secondDerivativeEdgeDetector(bluredImg) #Uncomment to use second Derivative Edge Detector
    # ImgEdge, direction = prewitt(bluredImg) #Uncomment to use prewitt Edge Detector

    # Non-maxima suppression (thinning Edges)
    NonMaximaImg = NonMaxima_suppression(ImgEdge, direction)

    # Highlight Strong Edges and ignore irrevelant edges using 2 thresholds
    double_thresh_img = double_threshold(NonMaximaImg)

    # Deal with weak edges
    final_img = Edge_Linking(double_thresh_img)

    # Show Image Processing Steps
    cv2.imshow("Original image", img)
    cv2.imshow("1 blured Image", bluredImg)
    cv2.imshow("2 Edge Image", ImgEdge)
    cv2.imshow("3 non-maxima Image ", NonMaximaImg)
    cv2.imshow("4 double thresh Image ", double_thresh_img)

    return final_img


# Main


def main():
    print("This might take some time please be patient ...")
    # Current Directory
    current_dir = os.getcwd()

    # Read Image from file_path

    file_path = ""  # insert image path here
    img = cv2.imread(file_path, 0)

    # Apply Canny Edge Detector
    final_img = Canny(img)

    # Show Final image
    cv2.imshow("5 final image", final_img)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
