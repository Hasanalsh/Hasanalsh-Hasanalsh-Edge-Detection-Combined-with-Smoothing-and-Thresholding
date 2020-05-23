import numpy as np
import cv2
from matplotlib import pyplot as plt

if __name__ == '__main__':
    image = 'Fig0235(c)(kidney_original).tif'     # image file name
    kernel_smooth = np.ones((5, 5), np.float32) / 25 # kernel matrix for Averaging smoothing
    kernel_edge = np.array([[-2, 0, 2],
                            [-2, 1, 2],
                            [-2, 0, 2]])             # kernel matrix for Sobel
    T = 180                                          # Thresholding value

    img = cv2.imread(image, 0)   # read image as gray

    img = cv2.filter2D(img, -1, kernel_smooth, borderType=cv2.BORDER_CONSTANT)          # Averaging smoothing
    img_filter = cv2.filter2D(img, -1, kernel_edge, borderType=cv2.BORDER_CONSTANT)     # edge detection using Sobel

    # Get threshold result with all threshold modes
    _, img_threshold1 = cv2.threshold(img_filter, T, 255, cv2.THRESH_BINARY)        # threshold with binary mode
    _, img_threshold2 = cv2.threshold(img_filter, T, 255, cv2.THRESH_BINARY_INV)    # threshold with binary_inv mode
    _, img_threshold3 = cv2.threshold(img_filter, T, 255, cv2.THRESH_TRUNC)         # threshold with trunc mode
    _, img_threshold4 = cv2.threshold(img_filter, T, 255, cv2.THRESH_TOZERO)        # threshold with tozero mode
    _, img_threshold5 = cv2.threshold(img_filter, T, 255, cv2.THRESH_TOZERO_INV)    # threshold with tozero_inv mode

    # Show the threshold results
    plt.subplot(2, 3, 1), plt.imshow(img_filter, cmap='gray')       # plot edge detection image
    plt.title('Sobel'), plt.xticks([]), plt.yticks([])              # title and axis
    plt.subplot(2, 3, 2), plt.imshow(img_threshold1, cmap='gray')   # plot threshold image with binary mode
    plt.title('BINARY'), plt.xticks([]), plt.yticks([])             # title and axis
    plt.subplot(2, 3, 3), plt.imshow(img_threshold2, cmap='gray')   # plot threshold image with binary_inv mode
    plt.title('BINARY_INV'), plt.xticks([]), plt.yticks([])         # title and axis
    plt.subplot(2, 3, 4), plt.imshow(img_threshold3, cmap='gray')   # plot threshold image with trunc mode
    plt.title('TRUNC'), plt.xticks([]), plt.yticks([])              # title and axis
    plt.subplot(2, 3, 5), plt.imshow(img_threshold4, cmap='gray')   # plot threshold image with tozero mode
    plt.title('TOZERO'), plt.xticks([]), plt.yticks([])             # title and axis
    plt.subplot(2, 3, 6), plt.imshow(img_threshold5, cmap='gray')   # plot threshold image with tozero_inv mode
    plt.title('TOZERO_INV'), plt.xticks([]), plt.yticks([])         # title and axis
    plt.show()      # show plot

    # Get the contours and select the max one
    contours, hierarchy = cv2.findContours(img_threshold1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # get the contours from the binary threshold image
    img_black = np.zeros(img.shape)     # create empty black image same size as img
    if len(contours) > 0:               # if the count of the contours > 0 (if there are contour) then...
        c = max(contours, key=cv2.contourArea)      # get the max area contour from the contours
        cv2.drawContours(img_black, [c], -1, 255, thickness=-1)     # draw the max contour to the black image

    # Display the final result
    plt.subplot(1, 2, 1), plt.imshow(img_threshold1, cmap='gray')   # plot binary threshold image
    plt.title('Binary Threshold'), plt.xticks([]), plt.yticks([])   # title and axis
    plt.subplot(1, 2, 2), plt.imshow(img_black, cmap='gray')        # plot max contour image
    plt.title('Final'), plt.xticks([]), plt.yticks([])              # title and axis
    plt.show()