import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

#SURF,BREIF,ORB
'''
img = cv2.imread("D:\Depot\Depot_GraduateSchool\IndustrialVision\scenetext01.jpg", cv2.IMREAD_COLOR)

surf = cv2.xfeatures2d.SURF_create(10000)
surf.setExtended(True)
surf.setNOctaves(3)
surf.setNOctaveLayers(10)
surf.setUpright(False)

keyPoints, descriptors = surf.detectAndCompute(img, None)

show_img = cv2.drawKeypoints(img, keyPoints, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('SURF descriptors', show_img)
cv2.waitKey()
cv2.destroyAllWindows()

brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(32, True)
keyPoints, descriptors = brief.compute(img, keyPoints)

show_img = cv2.drawKeypoints(img, keyPoints, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Brief descriptors', show_img)
cv2.waitKey()
cv2.destroyAllWindows()

orb = cv2.ORB_create()
orb.setMaxFeatures(200)

keyPoints = orb.detect(img, None)
keyPoints, descriptors = orb.compute(img, keyPoints)

show_img = cv2.drawKeypoints(img, keyPoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('ORB descriptors', show_img)
cv2.waitKey()
cv2.destroyAllWindows()
'''

#Finding correspondences between descriptors
'''
def video_keypoints(matcher, cap=cv2.VideoCapture("D:/Depot/Depot_GraduateSchool/IndustrialVision/traffic.mp4"), detector=cv2.ORB_create(40)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        status_cap, frame = cap.read()
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        if not status_cap:
            break
        if(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1 ) % 40 == 0:
            key_frame = np.copy(frame)
            key_points_1, descriptiors_1 = detector.detectAndCompute(frame, None)
        else:
            key_points_2, descriptiors_2 = detector.detectAndCompute(frame, None)
            matches = matcher.match(descriptiors_2, descriptiors_1)
            frame = cv2.drawMatches(frame, key_points_2, key_frame, key_points_1, matches, None,
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS |
                                    cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('Ketpoints matching', frame)
        if cv2.waitKey(300) == 27:
            break

    cv2.destroyAllWindows()


bf_matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING2, True)
video_keypoints(bf_matcher)

flann_kd_matcher = cv2.FlannBasedMatcher()
video_keypoints(flann_kd_matcher, detector=cv2.xfeatures2d.SURF_create(20000))

FLANN_INDEX_LSH = 6
index_params = dict(algorithm = FLANN_INDEX_LSH, table_number=20, key_size=15, multi_probe_lavel=2)
search_params = dict(checks=10)

flann_kd_matcher = cv2.FlannBasedMatcher(index_params, search_params)
video_keypoints(flann_kd_matcher)

FLANN_INDEX_COMPOSITE = 3
index_params = dict(algorithm=FLANN_INDEX_COMPOSITE, trees=16)
search_params = dict(checks=10)

flann_kd_matcher = cv2.FlannBasedMatcher(index_params, search_params)
video_keypoints(flann_kd_matcher, detector=cv2.xfeatures2d.SURF_create(20000))

'''
#Feature matching with consistency check and ratio test
'''
img0 = cv2.imread("D:/Depot/Depot_GraduateSchool/IndustrialVision/lena.png")
M=np.array( [[math.cos(np.pi/12), -math.sin(np.pi/12), 0],
             [math.sin(np.pi/12), math.cos(np.pi/12), 0],
             [0,0,1]])
Moff = np.eye(3)
Moff[0,2] = -img0.shape[1]/2
Moff[1,2] = -img0.shape[0]/2
print(np.linalg.inv(Moff)@M@Moff)
img1 = cv2.warpPerspective(img0, np.linalg.inv(Moff)@M@Moff,
                           (img0.shape[1], img0.shape[0]), borderMode=cv2.BORDER_REPLICATE)
cv2.imwrite("D:/Depot/Depot_GraduateSchool/IndustrialVision/lena_rotated001.png", img1)

img0 = cv2.imread("D:/Depot/Depot_GraduateSchool/IndustrialVision/lena.png", cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread("D:/Depot/Depot_GraduateSchool/IndustrialVision/lena_rotated001.png", cv2.IMREAD_GRAYSCALE)

detector = cv2.ORB_create(100)
kps0, fea0 = detector.detectAndCompute(img0, None)
kps1, fea1 = detector.detectAndCompute(img1, None)

matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, False)
matches01 = matcher.knnMatch(fea0, fea1, k=2)
matches10 = matcher.knnMatch(fea1, fea0, k=2)

def ratio_test(matches, ratio_thr):
    good_matches = []
    for m in matches:
        ratio = m[0].distance  / m[1].distance
        if ratio < ratio_thr:
            good_matches.append(m[0])
    return good_matches

RATIO_THR = 0.9
good_matches01 = ratio_test(matches01, RATIO_THR)
good_matches10 = ratio_test(matches10, RATIO_THR)

good_matches10_ = { (m.trainIdx, m.queryIdx) for m in good_matches10}
final_matches = [ m for m in good_matches01 if (m.queryIdx, m.trainIdx) in good_matches10_]

dbg_img = cv2.drawMatches(img0, kps0, img1, kps1, final_matches, None)
plt.figure()
plt.imshow(dbg_img[:,:,[2,1,0]])
plt.tight_layout()
plt.show()
'''

#Model based fitting using RANSAC
''''''

#BoW model for global image descriptor
''''''
matplotlib.rc('font', size=18)

img0 = cv2.imread("D:/Depot/Depot_GraduateSchool/IndustrialVision/people.jpg", cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread("D:/Depot/Depot_GraduateSchool/IndustrialVision/face.jpg", cv2.IMREAD_GRAYSCALE)

detector = cv2.ORB_create(500)
_, fea0 = detector.detectAndCompute(img0, None)
_, fea1 = detector.detectAndCompute(img1, None)
desrc_type = fea0.dtype

bow_trainer = cv2.BOWKMeansTrainer(50)
bow_trainer.add(np.float32(fea0))
bow_trainer.add(np.float32(fea1))
vocab = bow_trainer.cluster().astype(desrc_type)

bow_descr = cv2.BOWImgDescriptorExtractor(detector, cv2.BFMatcher(cv2.NORM_HAMMING))
bow_descr.setVocabulary(vocab)

img = cv2.imread("D:/Depot/Depot_GraduateSchool/IndustrialVision/lena.png", cv2.IMREAD_GRAYSCALE)
kps = detector.detect(img, None)
descr = bow_descr.compute(img, kps)

plt.figure(figsize = (10,8))
plt.title('image Bow descriptor')
plt.bar(np.arange(len(descr[0])), descr[0])
plt.xlabel('vocabulary element')
plt.ylabel('frequency')
plt.tight_layout()
plt.show()