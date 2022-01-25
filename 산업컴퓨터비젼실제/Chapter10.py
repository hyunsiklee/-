import cv2
import numpy as np
import glob
import os


'''
pattern_size = (10, 7)
samples = []
file_list = os.listdir('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/pinhole_calib')
img_file_list = [file for file in file_list if file.startswith('img')]

for filename in img_file_list:
    frame = cv2.imread(os.path.join('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/pinhole_calib', filename))
    res, corners = cv2.findChessboardCorners(frame, pattern_size)

    img_show = np.copy(frame)
    cv2.drawChessboardCorners(img_show, pattern_size, corners, res)
    cv2.putText(img_show, 'Samples capture: %d' % len(samples), (0, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),2)
    cv2.imshow('chessboard', img_show)

    wait_time = 0 if res else 30
    k= cv2.waitKey(wait_time)

    if k == ord('s') and res:
        samples.append( (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), corners))
    elif k == 27:
        break

cv2.destroyAllWindows()

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

for i in range(len(samples)):
    img, corners = samples[i]
    corners = cv2.cornerSubPix(img, corners, (10, 10), (-1,-1), criteria)

pattern_points = np.zeros((np.prod(pattern_size),3), np.float32)
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1,2)

images, corners = zip(*samples)

pattern_points = [pattern_points]*len(corners)

rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
    pattern_points, corners, images[0].shape, None, None)

np.save('camera_mat.npy', camera_matrix)
np.save('dist_coefs.npy', dist_coefs)

print(np.load('camera_mat.npy'))
print(np.load('dist_coefs.npy'))
'''


'''
pattern_size = (8, 6)
samples = []
file_list = os.listdir('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/fisheyes')
img_file_list = [file for file in file_list if file.startswith('Fisheye1_')]

for filename in img_file_list:
    frame = cv2.imread(os.path.join('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/fisheyes', filename))
    res, corners = cv2.findChessboardCorners(frame, pattern_size)

    res, corners = cv2.findChessboardCorners(frame, pattern_size)

    img_show = np.copy(frame)
    cv2.drawChessboardCorners(img_show, pattern_size, corners, res)
    cv2.putText(img_show, 'Samples capture: %d' % len(samples), (0, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow('chessboard', img_show)

    wait_time = 0 if res else 30
    k = cv2.waitKey(wait_time)

    if k == ord('s') and res:
        samples.append((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), corners))
    elif k == 27:
        break

cv2.destroyAllWindows()

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

for i in range(len(samples)):
    img, corners = samples[i]
    corners = cv2.cornerSubPix(img, corners, (10, 10), (-1, -1), criteria)

pattern_points = np.zeros((1, np.prod(pattern_size),3), np.float32)
pattern_points[0,:,:2] = np.indices(pattern_size).T.reshape(-1,2)

images, corners = zip(*samples)

pattern_points = [pattern_points]*len(corners)

print(len(pattern_points), pattern_points[0].shape, pattern_points[0].dtype)
print(len(corners), corners[0].shape, corners[0].dtype)

rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.fisheye.calibrate(
    pattern_points, corners, images[0].shape, None, None)

np.save('camera_mat.npy', camera_matrix)
np.save('dist_coefs.npy', dist_coefs)

print(np.load('camera_mat.npy'))
print(np.load('dist_coefs.npy'))
'''

'''
import cv2
import numpy as np
import os

pattern_size = (8, 6)
samples = []

file_list = os.listdir('D:/Depot/Depot_GraduateSchool/IndustrialVision/data//fisheyes')
img_file_list = [file for file in file_list if file.startswith('Fisheye2_')]

for filename in img_file_list:

    frame = cv2.imread(os.path.join('D:/Depot/Depot_GraduateSchool/IndustrialVision/data//fisheyes', filename))
    res, corners = cv2.findChessboardCorners(frame, pattern_size)

    img_show = np.copy(frame)
    cv2.drawChessboardCorners(img_show, pattern_size, corners, res)
    cv2.putText(img_show, 'Samples captured: %d' % len(samples), (0, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow('chessboard', img_show)

    wait_time = 0 if res else 30
    k = cv2.waitKey(wait_time)

    if k == ord('s') and res:
        samples.append((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), corners))
    elif k == 27:
        break

cv2.destroyAllWindows()

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

for i in range(len(samples)):
    img, corners = samples[i]
    corners = cv2.cornerSubPix(img, corners, (10, 10), (-1, -1), criteria)

pattern_points = np.zeros((1, np.prod(pattern_size), 3), np.float32)
pattern_points[0, :, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

images, corners = zip(*samples)

pattern_points = [pattern_points] * len(corners)

print(len(pattern_points), pattern_points[0].shape, pattern_points[0].dtype)
print(len(corners), corners[0].shape, corners[0].dtype)

camera_matrix = np.zeros((3, 3))
dist_coefs = np.zeros([1, 4])
xi = np.zeros(1)
rvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(len(corners))]
tvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(len(corners))]

rms = cv2.omnidir.calibrate(
    pattern_points, corners, images[0].shape, camera_matrix, xi, dist_coefs, cv2.omnidir.RECTIFY_PERSPECTIVE,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.000001), rvecs, tvecs)

np.save('camera_mat.npy', camera_matrix)
np.save('dist_coefs.npy', dist_coefs)

print(np.load('camera_mat.npy'))
print(np.load('dist_coefs.npy'))

img = cv2.imread('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/fisheyes/Fisheye2_2.jpg')
new_camera_matrix = np.copy(camera_matrix)
new_camera_matrix[0, 0] = new_camera_matrix[0, 0] / 3
new_camera_matrix[1, 1] = new_camera_matrix[1, 1] / 3
print(new_camera_matrix)
print(camera_matrix)
undistorted = np.zeros((640, 480, 3), np.uint8)
undistorted = cv2.omnidir.undistortImage(img, camera_matrix, dist_coefs, xi, cv2.omnidir.RECTIFY_PERSPECTIVE,
                                         undistorted, new_camera_matrix)

cv2.imshow("undistorted", undistorted)
cv2.waitKey(0)
'''

'''
PATTERN_SIZE = (9,6)
left_imgs = list(sorted(glob.glob('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/stereo/case1/left*.png')))
right_imgs = list(sorted(glob.glob('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/stereo/case1/right*.png')))
assert len(left_imgs) == len(right_imgs)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
left_pts, right_pts = [], []
img_size = None

for left_img_path, right_img_path in zip(left_imgs, right_imgs):
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
    if img_size is  None:
        img_size = (left_img.shape[1], left_img.shape[0])

    res_left, corners_left = cv2.findChessboardCorners(left_img, PATTERN_SIZE)
    res_right, corners_right = cv2.findChessboardCorners(right_img, PATTERN_SIZE)

    corners_left = cv2.cornerSubPix(left_img, corners_left, (10, 10), (-1, -1), criteria)
    corners_right = cv2.cornerSubPix(right_img, corners_right, (10, 10), (-1, -1), criteria)

    left_pts.append(corners_left)
    right_pts.append(corners_right)

pattern_points = np.zeros((np.prod(PATTERN_SIZE), 3), np.float32)
pattern_points[:,:2] = np.indices(PATTERN_SIZE).T.reshape(-1,2)
pattern_points = [pattern_points] * len(left_imgs)

err, Kl, Dl, Kr, Dr, R, T, E, F = cv2.stereoCalibrate(
    pattern_points, left_pts, right_pts, None, None, None, None, img_size, flags=0)

print('Left camera :')
print(Kl)
print('Left camera distortion:')
print(Dl)
print('Right camera :')
print(Kr)
print('Right camera distortion:')
print(Dr)
print('Rotation matrix :')
print(R)
print('Translation :')
print(T)

np.save('stereo.npy', {'Kl':Kl,'Dl':Dl,'Kr':Kr,'Dr':Dr,'R':R,'T':T,'E':E,'F':F,
                       'image_size':img_size,'left_pts':left_pts,'right_pts':right_pts,})
'''

'''
camera_matrix = np.load('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/pinhole_calib/camera_mat.npy')
dist_coefs = np.load('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/pinhole_calib/dist_coefs.npy')

img = cv2.imread('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/pinhole_calib/img_00.png')
pattern_size = (10, 7)
res, corners = cv2.findChessboardCorners(img, pattern_size)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
corners = cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                           corners, (10, 10), (-1, -1), criteria)

h_corners = cv2.undistortPoints(corners, camera_matrix, dist_coefs)
h_corners = np.c_[h_corners.squeeze(), np.ones(len(h_corners))]

img_pts,_ = cv2.projectPoints(h_corners, (0,0,0), (0,0,0), camera_matrix, None)

for c in corners:
    cv2.circle(img, tuple(c[0]), 10, (0,255,0),2)

for c in img_pts.squeeze().astype(np.float32):
    cv2.circle(img, tuple(c), 5, (0,0,255), 2)

cv2.imshow('undistorted corners', img)
cv2.waitKey()
cv2.destroyAllWindows()

img_pts,_ = cv2.projectPoints(h_corners, (0,0,0), (0,0,0), camera_matrix, dist_coefs)

for c in img_pts.squeeze().astype(np.float32):
    cv2.circle(img, tuple(c), 2, (255,255,0), 2)

cv2.imshow('reprojected corners', img)
cv2.waitKey()
cv2.destroyAllWindows()
'''

camera_matrix = np.load('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/pinhole_calib/camera_mat.npy')
dist_coefs = np.load('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/pinhole_calib/dist_coefs.npy')
img = cv2.imread('D:/Depot/Depot_GraduateSchool/IndustrialVision/data/pinhole_calib/img_00.png')

cv2.imshow('Original image', img)

ud_img = cv2.undistort(img, camera_matrix, dist_coefs)
cv2.imshow('Undistorted image1', ud_img)

opt_cam_mat, valid_roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, img.shape[:2][::-1],0)
ud_img = cv2.undistort(img, camera_matrix, dist_coefs, None, opt_cam_mat)
cv2.imshow('Undistorted image2', ud_img)

cv2.waitKey()
cv2.destroyAllWindows()


