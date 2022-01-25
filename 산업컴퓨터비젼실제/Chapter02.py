import cv2, math, numpy as np
import matplotlib.pyplot as plt
''' 영역 채우기
image_Fill = np.full((480, 640, 3), 255, np.uint8)
cv2.imshow('White', image)
cv2.waitKey()
cv2.destroyAllWindows()

image_Fill = np.full((480, 640, 3), (0,0,255), np.uint8)
cv2.imshow('Red', image_Fill)
cv2.waitKey()
cv2.destroyAllWindows()

image_Fill.fill(0)
cv2.imshow('Black', image_Fill)
cv2.waitKey()
cv2.destroyAllWindows()

image_Fill[240,160] = image_Fill[240,320] = image_Fill[240,480] = (0,255,255)
cv2.imshow('Black with white pixels', image_Fill)
cv2.waitKey()
cv2.destroyAllWindows()

image_Fill[:,:,0] = 255
cv2.imshow('Blue with white pixels', image_Fill)
cv2.waitKey()
cv2.destroyAllWindows()

image_Fill[:,320,:] = 255
cv2.imshow('Blue with white lines', image_Fill)
cv2.waitKey()
cv2.destroyAllWindows()

image_Fill[50:400,150:300,1] = 255
cv2.imshow('Image', image_Fill)
cv2.waitKey()
cv2.destroyAllWindows()
'''

''' 색좌표 변환
colorSpace = cv2.imread("D:\Depot\Depot_GraduateSchool\IndustrialVision\lena.png").astype(np.float32)/255
print('shape: ',colorSpace.shape)
print('Data type: ',colorSpace.dtype)
cv2.imshow('Original image', colorSpace)

graySpace = cv2.cvtColor(colorSpace, cv2.COLOR_BGR2GRAY)
print('Convert grayscale')
print('shape: ',graySpace.shape)
print('Data type: ',graySpace.dtype)
cv2.imshow('Gray scale image', graySpace)

hsv = cv2.cvtColor(colorSpace, cv2.COLOR_BGR2HSV)
print('Convert HSV')
print('shape: ',graySpace.shape)
print('Data type: ',graySpace.dtype)
hsv[:,:,2] *= 2
from_HSV = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
print('Convert Back to BGR to HSV')
print('shape: ',from_HSV.shape)
print('Data type: ',from_HSV.dtype)
cv2.imshow('from HSV', from_HSV)
cv2.waitKey()
cv2.destroyAllWindows()
'''

''' 히스토그램 및 평활화
grey = cv2.imread("D:\Depot\Depot_GraduateSchool\IndustrialVision\lena.png", 0)
cv2.imshow('Original grey', grey)
cv2.waitKey()
cv2.destroyAllWindows()

hist, bins = np.histogram(grey, 256, [0,255])
#plt.xlabel('Pixel value')
#plt.show()

grey_eq = cv2.equalizeHist(grey)
hist2, bins2 = np.histogram(grey_eq, 256, [0,255])

plt.fill(hist)
plt.fill_between(range(256), hist2, 0)
plt.tight_layout()
plt.show()

#plt.xlabel('Pixel value')
#plt.show()
plt.figure(figsize=[8,4])
plt.subplot(121)
plt.title('Original histogram')
plt.fill(hist)
plt.subplot(122)
plt.title('Equalized histogram')
plt.fill_between(range(256), hist2, 0)
plt.tight_layout()
plt.show()

plt.figure(figsize=[8,4])
plt.subplot(121)
plt.axis('off')
plt.title('Original grey')
plt.imshow(grey,cmap='gray')
plt.subplot(122)
plt.axis('off')
plt.title('Equalized grey')
plt.imshow(grey_eq, cmap='gray')
plt.tight_layout()
plt.show()
#cv2.imshow('Original grey', grey)
#cv2.imshow('Equalized grey', grey_eq)
#cv2.waitKey()
#cv2.destroyAllWindows()

# Color equalized histogram
color = cv2.imread("D:\Depot\Depot_GraduateSchool\IndustrialVision\lena.png")
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
hsv[...,2] = cv2.equalizeHist(hsv[..., 2])
color_eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
plt.figure(figsize=[8,4])
plt.subplot(121)
plt.axis('off')
plt.title('Original color')
plt.imshow(color[:,:,[2,1,0]])
plt.subplot(122)
plt.axis('off')
plt.title('Equalized color')
plt.imshow(color_eq[:,:,[2,1,0]])
plt.tight_layout()
plt.show()
#cv2.imshow('Original color', color)
#cv2.imshow('Equalized color', color_eq)
#cv2.waitKey()
#cv2.destroyAllWindows()
'''
#filter
image = cv2.imread("D:\Depot\Depot_GraduateSchool\IndustrialVision\lena.png").astype(np.float32)/255
noised = (image + 0.2 * np.random.rand(*image.shape).astype(np.float32))
noised = noised.clip(0,1)
gauss_blur = cv2.GaussianBlur(noised, (7,7), 0)
median_blur = cv2.medianBlur((noised * 255).astype(np.uint8), 7)
bolat = cv2.bilateralFilter(noised, -1, 0.3, 10)

plt.figure(figsize=(9,9))
plt.subplot(221)
plt.axis('off')
plt.title('Noised')
plt.imshow(noised[:,:,[2,1,0]])
plt.subplot(222)
plt.axis('off')
plt.title('Gaussian Blur')
plt.imshow(gauss_blur[:,:,[2,1,0]])
plt.subplot(223)
plt.axis('off')
plt.title('Median Blur')
plt.imshow(median_blur[:,:,[2,1,0]])
plt.subplot(224)
plt.axis('off')
plt.title('bilateralFilter')
plt.imshow(bolat[:,:,[2,1,0]])
plt.tight_layout()
plt.show()

#plt.imshow(noised[:,:,[2,1,0]])
#plt.show()
#gauss_blur = cv2.GaussianBlur(noised, (7,7), 0)
#plt.imshow(gauss_blur[:,:,[2,1,0]])
#plt.show()
#median_blur = cv2.medianBlur((noised * 255).astype(np.uint8), 7)
#plt.imshow(median_blur[:,:,[2,1,0]])
#plt.show()
#bolat = cv2.bilateralFilter(noised, -1, 0.3, 10)
#plt.imshow(bolat[:,:,[2,1,0]])
#plt.show()

'''
#sobel
dx = cv2.Sobel(grey, cv2.CV_32F, 1, 0)
dy = cv2.Sobel(grey, cv2.CV_32F, 0, 1)
plt.figure(figsize=(8,3))
plt.subplot(131)
plt.axis('off')
plt.title('image')
plt.imshow(grey, cmap='gray')
plt.subplot(132)
plt.axis('off')
plt.imshow(dx, cmap='gray')
plt.title(r'$\frac{dI}{dx}$')
plt.subplot(133)
plt.axis('off')
plt.title(r'$\frac{dI}{dy}$')
plt.imshow(dy, cmap='gray')
plt.tight_layout()
plt.show()

#filter2D
image_Colored = cv2.imread("D:\Depot\Depot_GraduateSchool\IndustrialVision\lena.png")
KSIZE = 11
ALPHA = 2
kernel = cv2.getGaussianKernel(KSIZE, 0)
kernel = -ALPHA * kernel@kernel.T
kernel[KSIZE//2, KSIZE//2] += 1 + ALPHA
filtered = cv2.filter2D(image_Colored, -1, kernel)
plt.figure(figsize=(8,4))
plt.subplot(121)
plt.axis('off')
plt.title('image')
plt.imshow(image_Colored[:,:,[2,1,0]])
plt.subplot(122)
plt.axis('off')
plt.title('filtered')
plt.imshow(filtered[:,:,[2,1,0]])
plt.tight_layout()
plt.show()'''

