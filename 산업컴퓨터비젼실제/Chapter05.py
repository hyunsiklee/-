import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("D:\Depot\Depot_GraduateSchool\IndustrialVision\lena.png").astype(np.float32) / 255
assert image is not None

#K-Mean
#Watershed
#GrabCut
#CannyEdgeDetection
#hough Transform