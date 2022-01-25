import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size':20})
'''
P1 = np.eye(3,4, dtype=np.float32)
P2 = np.eye(3,4, dtype=np.float32)
P2[0,3] = -1

N = 5
points3d = np.empty((4, N), np.float32)
points3d[:3, :] = np.random.randn(3, N)
points3d[3,:] = 1

points1 = P1 @ points3d
points1 = points1[:2,:] / points1[2,:1]
points1[:2,:] += np.random.randn(2,N)*1e-2

points2 = P2 @ points3d
points2 = points2[:2,:] / points2[2,:1]
points2[:2,:] += np.random.randn(2,N)*1e-2

points3d_reconstr = cv2.triangulatePoints(P1, P2, points1, points2)
points3d_reconstr /= points3d_reconstr[3, :]

print('Original points')
print(points3d[:3].T)
print('Reconstructed points')
print(points3d_reconstr[:3].T)
'''

'''
camera_matrix = np.load('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/pinhole_calib/camera_mat.npy')
dist_coefs = np.load('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/pinhole_calib/dist_coefs.npy')
img = cv2.imread('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/pinhole_calib/img_00.png')

pattern_size = (10,7)
res, corners = cv2.findChessboardCorners(img, pattern_size)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
corners = cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), corners, (10, 10), (-1,-1), criteria)

pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1,2)

ret, rvec, tvec = cv2.solvePnP(pattern_points, corners, camera_matrix, dist_coefs, None, None, False, cv2.SOLVEPNP_ITERATIVE)

img_points, _ = cv2.projectPoints(pattern_points, rvec, tvec, camera_matrix, dist_coefs)
for c in img_points.squeeze():
    cv2.circle(img, tuple(c), 10, (0,255,0),2)

cv2.imshow('points', img)
cv2.waitKey()
cv2.destroyAllWindows()
'''

'''
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

data = np.load('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/stereo/case1/stereo.npy').item()
Kl, Dl, Kr, Dr, R, T, img_size = data['Kl'], data['Dl'], data['Kr'], data['Dr'], \
                                 data['R'], data['T'], data['img_size']

left_img = cv2.imread('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/stereo/case1/left14.png')
right_img = cv2.imread('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/stereo/case1/right14.png')

R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(Kl, Dl, Kr, Dr, img_size, R, T)

xmap1, ymap1 = cv2.initUndistortRectifyMap(Kl, Dl, R1, Kl, img_size, cv2.CV_32FC1)
xmap2, ymap2 = cv2.initUndistortRectifyMap(Kr, Dr, R2, Kr, img_size, cv2.CV_32FC1)

left_img_rectified = cv2.remap(left_img, xmap1, ymap1, cv2.INTER_LINEAR)
right_img_rectified = cv2.remap(right_img, xmap2, ymap2, cv2.INTER_LINEAR)

plt.figure(0, figsize=(12,10))
plt.subplot(221)
plt.title('left original')
plt.imshow(left_img, cmap='gray')
plt.subplot(222)
plt.title('right original')
plt.imshow(right_img, cmap='gray')
plt.subplot(223)
plt.title('left rectified')
plt.imshow(left_img_rectified, cmap='gray')
plt.subplot(224)
plt.title('right rectified')
plt.imshow(right_img_rectified, cmap='gray')
plt.tight_layout()
plt.show()
'''

'''
matplotlib.rcParams.update({'font.size':20})
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

data = np.load('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/stereo/case1/stereo.npy').item()
Kl, Kr, Dl, Dr, left_pts, right_pts, E_from_stereo, F_From_stereo = \
data['Kl'], data['Kr'], data['Dl'], data['Dr'], data['left_pts'], data['right_pts'], data['E'],data['F']

left_pts = np.vstack(left_pts)
right_pts = np.vstack(right_pts)

left_pts = cv2.undistortPoints(left_pts, Kl, Dl, P=Kl)
right_pts = cv2.undistortPoints(right_pts, Kr, Dr, P=Kr)

F, mask = cv2.findFundamentalMat(left_pts, right_pts, cv2.FM_LMEDS)
E=Kr.T @ F @ Kl

print('Fundamental matrix:')
print(F)
print('Essentail matrix:')
print(E)
'''

'''
matplotlib.rcParams.update({'font.size':20})
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

data = np.load('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/stereo/case1/stereo.npy').item()
E = data['E']

R1, R2, T = cv2.decomposeEssentialMat(E)

print('Rotation 1: ')
print(R1)
print('Rotation 2: ')
print(R2)
print('Translation: ')
print(T)
'''

''' '''
left_img = cv2.imread('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/stereo/left.png')
right_img = cv2.imread('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/stereo/right.png')

stereo_bm = cv2.StereoBM_create(32)
dispmap_bm = stereo_bm.compute(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right_img,cv2.COLOR_BGR2GRAY))

stereo_sgbm = cv2.StereoSGBM_create(0, 32)
dispmap_sgbm = stereo_sgbm.compute(left_img, right_img)

plt.figure(0, figsize=(12,10))
plt.subplot(221)
plt.title('left')
plt.imshow(left_img[:,:,[2,1,0]])
plt.subplot(222)
plt.title('right')
plt.imshow(right_img[:,:,[2,1,0]])
plt.subplot(223)
plt.title('BM')
plt.imshow(dispmap_bm, cmap='gray')
plt.subplot(224)
plt.title('SGBM')
plt.imshow(dispmap_sgbm, cmap='gray')
plt.tight_layout()
plt.show()
