import numpy as np
import cv2

def EdgeDetection(img_path):
    # Load the image
    img = cv2.imread(img_path)
    
    # Resize the image to a fixed size
    img = cv2.resize(img, (550, 440))
    
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Pad the image with zeros around the edges
    padded_img = np.pad(gray_img, ((1, 1), (1, 1)), 'constant')
    
    # Define Sobel filters for horizontal and vertical edges
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Create arrays to hold the horizontal and vertical edge information
    sobelx = np.zeros_like(gray_img, dtype=np.float32)
    sobely = np.zeros_like(gray_img, dtype=np.float32)
    
    # Loop through each pixel in the image
    for i in range(1, gray_img.shape[0]-1):
        for j in range(1, gray_img.shape[1]-1):
            # Extract the 3x3 pixel neighborhood around the current pixel
            neighborhood = np.array(padded_img[i-1:i+2, j-1:j+2])
            
            # Convolve the neighborhood with the Sobel filters
            sobelx[i,j] = np.sum(np.multiply(sobel_x, neighborhood))
            sobely[i,j] = np.sum(np.multiply(sobel_y, neighborhood))
    
    # Compute the magnitude of the edges using the horizontal and vertical edge information
    mag = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))
    
    # Normalize the magnitude to the range [0, 255] and convert to unsigned 8-bit integer
    mag = ((mag - np.min(mag)) / (np.max(mag) - np.min(mag)) * 255).astype(np.uint8)
    
    # Display the edge image
    cv2.imshow('Sobel Detection', mag)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function with the path to the input image
EdgeDetection('test3.png')