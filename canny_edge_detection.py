import numpy as np
from scipy import ndimage
from scipy.ndimage import convolve
import cv2
import os

class cannyEdgeDetector:
    def __init__(self, imgs, img_names, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):
        self.imgs = imgs
        self.img_names = img_names
        self.imgs_final = []
        self.img_smoothed = None
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.lowThreshold = lowthreshold
        self.highThreshold = highthreshold
        
    # Apply Gaussian Filter to smooth the image and reduce noise
        
    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g
    
    # Gradient Calculation using Sobel Operator
    
    def sobel_filters(self, img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = ndimage.convolve(img, Kx)
        Iy = ndimage.convolve(img, Ky)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        return (G, theta)
    
    # Non-Maximum Suppression to thin the edges
    
    def non_max_suppression(self, img, D):
        M, N = img.shape
        Z = np.zeros((M,N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255

                   #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img[i, j+1]
                        r = img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    if (img[i,j] >= q) and (img[i,j] >= r):
                        Z[i,j] = img[i,j]
                    else:
                        Z[i,j] = 0

                except IndexError as e:
                    pass

        return Z
    
    # Double Thresholding to determine potential edges
    
    def threshold(self, img):
        highThreshold = img.max() * self.highThreshold
        lowThreshold = highThreshold * self.lowThreshold

        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)

        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)

        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return res
    
    # Edge Tracking by Hysteresis to finalize edges
    
    def hysteresis(self, img):
        M, N = img.shape
        weak = self.weak_pixel
        strong = self.strong_pixel
        out_img = np.copy(img)

        for i in range(1, M-1):
            for j in range(1, N-1):
                if (out_img[i,j] == weak):
                    try:
                        if ((out_img[i+1, j-1] == strong) or (out_img[i+1, j] == strong) or (out_img[i+1, j+1] == strong)
                            or (out_img[i, j-1] == strong) or (out_img[i, j+1] == strong)
                            or (out_img[i-1, j-1] == strong) or (out_img[i-1, j] == strong) or (out_img[i-1, j+1] == strong)):
                            out_img[i, j] = strong
                        else:
                            out_img[i, j] = 0
                    except IndexError as e:
                        pass

        return out_img
    
    def detect_and_save_steps(self):
        
        # Executes the Canny Edge Detection pipeline and saves intermediate matrices.
        self.imgs_final = []
        for i, img in enumerate(self.imgs):
            base_name = self.img_names[i]
            
            # Step 1: Gaussian Blur
            self.img_smoothed = convolve(img, self.gaussian_kernel(self.kernel_size, self.sigma))
            cv2.imwrite(f"{base_name}_1_smoothed.jpg", self.img_smoothed.astype(np.uint8))
            
            # Step 2: Sobel Filters (Gradient Calculation)
            self.gradientMat, self.thetaMat = self.sobel_filters(self.img_smoothed)
            cv2.imwrite(f"{base_name}_2_gradient.jpg", self.gradientMat.astype(np.uint8))
            
            # Step 3: Non-Maximum Suppression
            self.nonMaxImg = self.non_max_suppression(self.gradientMat, self.thetaMat)
            cv2.imwrite(f"{base_name}_3_non_max.jpg", self.nonMaxImg.astype(np.uint8))
            
            # Step 4: Double Threshold
            self.thresholdImg = self.threshold(self.nonMaxImg)
            cv2.imwrite(f"{base_name}_4_threshold.jpg", self.thresholdImg.astype(np.uint8))
            
            # Step 5: Hysteresis Tracking
            img_final = self.hysteresis(self.thresholdImg)
            cv2.imwrite(f"{base_name}_5_final.jpg", img_final.astype(np.uint8))
            
            self.imgs_final.append(img_final)

        return self.imgs_final

if __name__ == "__main__":
    # Define source dependencies
    image_paths = ["images/GS.jpg", "images/charlie.jpg"]
    loaded_images = []
    file_names = []
    
    # Read Grayscale Input
    for path in image_paths:
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            loaded_images.append(img)
            file_names.append(os.path.splitext(os.path.basename(path))[0])
        else:
            print(f"Error: Dependency {path} not found.")

    if loaded_images:
        # Initialize the procedural class with extracted matrices
        detector = cannyEdgeDetector(
            imgs=loaded_images, 
            img_names=file_names,
            sigma=1.4,          # Adjusted sigma for standard diffusion
            kernel_size=5, 
            lowthreshold=0.05, 
            highthreshold=0.15  # Regulated upper limit to constrain artifacts
        )
        
        # Execute processing pipeline and output sequential matrices
        detector.detect_and_save_steps()
        print("Algorithmic extraction complete. Sequential image representations saved.")

