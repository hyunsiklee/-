import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='D:\Depot\Depot_GraduateSchool\IndustrialVision\lena.png', help='Image path.')
params = parser.parse_args()
img = cv2.imread(params.path)

assert img is not None

print('read: {}'.format(params.path))
print('shape:', img.shape)
print('dtype:', img.dtype)

img = cv2.imread(params.path, cv2.IMREAD_GRAYSCALE)

assert img is not None

nX = (int)(img.shape[0] / 2)
nY = (int)(img.shape[1] / 2)
centerGV = img[nX,nY]
print('read: {} as gray scale'.format(params.path))
print('shape:', img.shape)
print('dtype:', img.dtype)
print('center:', centerGV)

img = cv2.imread(params.path)
cv2.namedWindow('Original')
cv2.imshow('Original', img)
cv2.waitKey(2000)

width, height = 128, 256
resized_img = cv2.resize(img, (width, height))
cv2.namedWindow('Resize_128, 256')
cv2.imshow('Resize_128, 256', resized_img)
cv2.waitKey(2000)

w_mult, h_mult = 0.25, 0.5
resized_img = cv2.resize(img, (0,0), resized_img, w_mult, h_mult)
cv2.namedWindow('Resize_0.25, 0.5')
cv2.imshow('Resize_0.25, 0.5', resized_img)
cv2.waitKey(2000)

w_mult, h_mult = 1.5, 1.5
resized_img = cv2.resize(img, (0,0), resized_img, w_mult, h_mult, cv2.INTER_CUBIC)
cv2.namedWindow('Resize_x2, x4')
cv2.imshow('Resize_x2, x4', resized_img)
cv2.waitKey(2000)

img_flip_along_x = cv2.flip(img, 0)
img_flip_along_x_along_y = cv2.flip(img_flip_along_x, 1)
img_flip_along_y = cv2.flip(img, -1)

cv2.namedWindow('Filp X')
cv2.imshow('Filp X', img_flip_along_x)
cv2.waitKey(2000)
cv2.namedWindow('Filp XY')
cv2.imshow('Filp XY', img_flip_along_x_along_y)
cv2.waitKey(2000)
cv2.namedWindow('Filp Y')
cv2.imshow('Filp Y', img_flip_along_y)
cv2.waitKey(2000)

