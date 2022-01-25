import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

'''
image = cv2.imread("D:\Depot\Depot_GraduateSchool\IndustrialVision\scenetext01.jpg", cv2.IMREAD_COLOR)
corners = cv2.cornerHarris(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)

corners = cv2.dilate(corners, None)

show_img = np.copy(image)
show_img [corners > 0.1 * corners.max()] = [0,0,255]

corners = cv2.normalize(corners, None , 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
show_img = np.hstack((show_img, cv2.cvtColor(corners, cv2.COLOR_GRAY2BGR)))

cv2.imshow('Harris corner setector', show_img)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()

fast = cv2.FastFeatureDetector_create(30, True, cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
kp = fast.detect(image)

show_img = np.copy(image)
for p in cv2.KeyPoint_convert(kp):
    cv2.circle(show_img, tuple(p), 2, (0, 255, 0), cv2.FILLED)

cv2.imshow('FAST corner setector', show_img)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()

fast.setNonmaxSuppression(False)
kp = fast.detect(image)

for p in cv2.KeyPoint_convert(kp):
    cv2.circle(show_img, tuple(p), 2, (0, 255, 0), cv2.FILLED)

cv2.imshow('FAST corner setector', show_img)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
'''

'''
img = cv2.imread("D:\Depot\Depot_GraduateSchool\IndustrialVision\lena.png", cv2.IMREAD_GRAYSCALE)
assert img is not None

corners = cv2.goodFeaturesToTrack(img, 100, 0.05, 10)

for c in corners:
    x, y = c[0]
    cv2.circle(img, (x,y), 5, 255, -1)

plt.figure(figsize = (8,8))
plt.imshow(img, cmap = 'gray')
plt.tight_layout()
plt.show()
'''

'''
img = cv2.imread("D:\Depot\Depot_GraduateSchool\IndustrialVision\scenetext01.jpg", cv2.IMREAD_COLOR)

fast = cv2.FastFeatureDetector_create(160, True, cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
keyPoints = fast.detect(img)

for kp in keyPoints:
    kp.size = 100*random.random()
    kp.angle = 360*random.random()

matches = []
for i in range(len(keyPoints)):
    matches.append(cv2.DMatch(i,i,1))

show_img = cv2.drawKeypoints(img, keyPoints, None, (255, 0, 255))

cv2.imshow('KetPoints', show_img)
cv2.waitKey()
cv2.destroyAllWindows()

show_img = cv2.drawKeypoints(img, keyPoints, None, (0, 255, 0),
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('KetPoints', show_img)
cv2.waitKey()
cv2.destroyAllWindows()

show_img = cv2.drawMatches(img, keyPoints, img, keyPoints, matches, None,
                             flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('matches', show_img)
cv2.waitKey()
cv2.destroyAllWindows()
'''

''' '''
img0 = cv2.imread("D:\Depot\Depot_GraduateSchool\IndustrialVision\lena.png", cv2.IMREAD_COLOR)
img1 = cv2.imread("D:\Depot\Depot_GraduateSchool\IndustrialVision\lena_Rotate.png", cv2.IMREAD_COLOR)
img1 = cv2.resize(img1, None, fx=0.75, fy=0.75)
img1 = np.pad(img1, ((64,)*2,(64,)*2,(0,)*2), 'constant', constant_values=0)
imgs_list = [img0, img1]

detector = cv2.xfeatures2d.SIFT_create(50)

for i in range(len(imgs_list)):
    ketpoints, descriptors = detector.detectAndCompute(imgs_list[i], None)

    imgs_list[i] = cv2.drawKeypoints(imgs_list[i], ketpoints, None, (0, 255, 0),
                                     flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('SIFT ketPonits', np.hstack(imgs_list))
cv2.waitKey()
cv2.destroyAllWindows()
