import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    image = 'Fig0235(c)(kidney_original).tif'       # image file name

    # Loading image
    img0 = cv2.imread(image)                        # read image as color
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)   # convert image color into grayscale

    # Remove noise with Gaussian
    img = cv2.GaussianBlur(gray, (3, 3), 0)         # Gaussian smoothing

    # Remove noise with Averaging
    kernel_smooth = np.ones((5, 5), np.float32) / 25    # Averaging kernel
    img_avg = cv2.filter2D(img, -1, kernel_smooth, borderType=cv2.BORDER_CONSTANT)  # Averaging smoothing

    # Show smoothing result
    plt.subplot(1, 3, 1), plt.imshow(gray, cmap='gray')     # draw gray original image
    plt.title('Original'), plt.xticks([]), plt.yticks([])   # title and axis
    plt.subplot(1, 3, 2), plt.imshow(img_avg, cmap='gray')  # draw averaging smoothing image
    plt.title('Averaging'), plt.xticks([]), plt.yticks([])  # title and axis
    plt.subplot(1, 3, 3), plt.imshow(img, cmap='gray')      # draw Gaussian smoothing image
    plt.title('Gaussian'), plt.xticks([]), plt.yticks([])   # title and axis
    plt.show()      # show plot

    # Sobel gradient using the masks in Fig. 10.14
    sobel_x = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # kernel of sobel x
    sobel_y1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])   # kernel of sobel y1
    sobel_y2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])   # kernel of sobel y2


    # Compute Sobel gradient with x & y axises and get binary image using threshold
    img_x = cv2.filter2D(img, -1, sobel_x, borderType=cv2.BORDER_CONSTANT)      # Sobel x filtering

    img_y1 = cv2.filter2D(img, -1, sobel_y1, borderType=cv2.BORDER_CONSTANT)  # Sobel y1 filtering
    img_y2 = cv2.filter2D(img, -1, sobel_y2, borderType=cv2.BORDER_CONSTANT)  # Sobel y2 filtering
    img_y = img_y1 + img_y2  # merge Sobel y1 and y2

    grad_x = img_x.astype(np.float32)   # convert Sobel x image into float32
    grad_y = img_y.astype(np.float32)   # convert Sobel y image into float32
    img_sobel = np.sqrt(np.power(grad_x, 2) + np.power(grad_y, 2))  # calculate Sobel using Eq. (10.2-20)
    img_sobel *= 255.0 / img_sobel.max()    # normalizing of ٍٍٍSobel result (make max value to 255)
    img_sobel = img_sobel.astype(np.uint8)  # convert the type into uint8
    T = 70
    _, dst_threshold = cv2.threshold(img_sobel, T, 255, cv2.THRESH_BINARY)  # thresholding using binary mode

    # Show calculated Sobel results of x and y.
    plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray')  # plot Gaussian smoothing image
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 2), plt.imshow(img_x, cmap='gray')  # plot Sobel_x image
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 3), plt.imshow(img_y, cmap='gray')  # plot Sobel_y image
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.show()  # show plot

    # Convolute with proper kernels with function
    sobelx = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # Sobel_x filtering
    sobely = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # Sobel_y filtering
    sobel = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)  # Sobel filtering combine of x and y

    # Show Sobel results with function
    plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray')  # plot Gaussian smoothing image
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 2), plt.imshow(sobelx, cmap='gray')  # plot Sobel_x image
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 3), plt.imshow(sobely, cmap='gray')  # plot Sobel_y image
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.show()  # show plot

    # Display Sobel and threshold results
    plt.subplot(1, 2, 1), plt.imshow(img_sobel, cmap='gray')    # plot Sobel image
    plt.title('Sobel X + Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(dst_threshold, cmap='gray')    # plot threshold image
    plt.title('Thresholding'), plt.xticks([]), plt.yticks([])
    plt.show()  # show plot