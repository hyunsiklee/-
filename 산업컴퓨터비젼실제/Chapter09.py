import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

#Optical flow & Panorama stitching

''' 
#Warping image using affine and perspective transformation
img = cv2.imread("D:\Depot\Depot_GraduateSchool\IndustrialVision\circlesgrid.png", cv2.IMREAD_COLOR)
show_img = np.copy(img)
selected_pts = []

def mouse_callback(event, x,y, flags, params):
    global selected_pts, show_img
    if event == cv2.EVENT_LBUTTONUP:
        selected_pts.append([x,y])
        cv2.circle(show_img, (x,y), 10, (0,255,0), 3)

def select_points(image, points_num):
    global selected_pts
    selected_pts = []
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)

    while True:
        cv2.imshow('image', image)
        k = cv2.waitKey(1)
        if k == 27 or len(selected_pts) == points_num:
            break

    cv2.destroyAllWindows()
    return np.array(selected_pts, dtype = np.float32)

show_img = np.copy(img)

src_pts = select_points(show_img, 3)
dst_pts = np.array( [[0,240], [0,0], [240,0]], dtype=np.float32)

affine_m = cv2.getAffineTransform(src_pts, dst_pts)
unwarped_img = cv2.warpAffine(img, affine_m, (240,240))

cv2.imshow('resilt', np.hstack((show_img, unwarped_img)))
k=cv2.waitKey()
cv2.destroyAllWindows()

inv_affine = cv2.invertAffineTransform(affine_m)
warped_img = cv2.warpAffine(unwarped_img, inv_affine, (320,240))
cv2.imshow('result', np.hstack((show_img, unwarped_img, warped_img)))
k=cv2.waitKey()
cv2.destroyAllWindows()

show_img = np.copy(img)
src_pts = select_points(show_img, 4)
dst_pts = np.array( [[0,240],[0,0],[240,0],[240,240]], dtype=np.float32)
perspective_m = cv2.getPerspectiveTransform(src_pts, dst_pts)
unwarped_img = cv2.warpPerspective(img, perspective_m, (240,240))
cv2.imshow('resilt', np.hstack((show_img, unwarped_img)))
k=cv2.waitKey()
cv2.destroyAllWindows()
'''

'''
#Remapping using arbitrary transformation
img = cv2.imread("D:\Depot\Depot_GraduateSchool\IndustrialVision\lena.png")

xmap = np.zeros((img.shape[1], img.shape[0]), np.float32)
ymap = np.zeros((img.shape[1], img.shape[0]), np.float32)
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        xmap[y,x] = x + 30 * math.cos(20 * x / img.shape[0])
        ymap[y,x] = y + 30 * math.sin(20 * y / img.shape[1])
    
remapped_img = cv2.remap(img, xmap, ymap , cv2.INTER_LINEAR, None, cv2.BORDER_REPLICATE)
'''
'''
#Tracking keypoints between frames Lucas-Kanade algorithm
video = cv2.VideoCapture("D:/Depot/Depot_GraduateSchool/IndustrialVision/traffic.mp4")
prev_pts = None
prev_gray_frame = None
tracks = None

while True:
    retval, frame = video.read()
    if not retval: break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_pts is not None:
        pts, status, errors = cv2.calcOpticalFlowPyrLK(
            prev_gray_frame, gray_frame, prev_pts, None, winSize=(15,15), maxLevel=5,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        good_pts = pts[status == 1]
        if tracks is None: tracks = good_pts
        else : tracks = np.vstack((tracks, good_pts))
        for p in tracks:
            cv2.circle(frame, (p[0], p[1]), 3, (0,255,0) , -1)
    else:
        pts = cv2.goodFeaturesToTrack(gray_frame, 500, 0.05, 10)
        pts = pts.reshape(-1,1,2)

    prev_pts = pts
    prev_gray_frame = gray_frame

    cv2.imshow('frame', frame)
    key = cv2.waitKey() & 0xff
    if key == 27:break
    if key == ord('c'):
        tracks = None
        prev_pts = None

cv2.destroyAllWindows()
'''

'''
#Dense optical flow between two frames
def display_flow(img, flow, stride=40):
    for index in np.ndindex(flow[::stride, ::stride].shape[:2]):
        pt1 = tuple(i*stride for i in index)
        delta = flow[pt1].astype(np.int32)[::1]
        pt2 = tuple(pt1 + 10 * delta)
        if 2 <= cv2.norm(delta) <= 10:
            cv2.arrowedLine(img, pt1[::-1], pt2[::-1],
                            (0,0,255),5,cv2.LINE_AA, 0, 0.4)

    norm_opt_flow = np.linalg.norm(flow, axis=2)
    norm_opt_flow = cv2.normalize(norm_opt_flow, None, 0, 1,
                                  cv2.NORM_MINMAX)

    cv2.imshow('Optical flow', img)
    cv2.imshow('Optical flow magnitude', norm_opt_flow)
    k = cv2.waitKey(1)

    if k == 27:
        return 1
    else:
        return 0

cap = cv2.VideoCapture("D:/Depot/Depot_GraduateSchool/IndustrialVision/traffic.mp4")
_,prev_frame = cap.read()

prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame = cv2.resize(prev_frame, (0,0), None , 0.5, 0.5)
init_flow = True

while True:
    status_cap, frame = cap.read()
    frame = cv2.resize(frame, (0,0), None, 0.5, 0.5)
    if not status_cap:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if init_flow:
        opt_flow = cv2.calcOpticalFlowFarneback(
            prev_frame, gray, None, 0.5, 5, 13, 10,
            5,1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        init_flow = False
    else:
        opt_flow = cv2.calcOpticalFlowFarneback(
            prev_frame, gray, opt_flow, 0.5, 5, 13,
            10, 5,1.1, cv2.OPTFLOW_USE_INITIAL_FLOW)

    prev_frame = np.copy(gray)

    if display_flow(frame, opt_flow):
        break

cv2.destroyAllWindows()
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
_, prev_frame = cap.read()

prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame = cv2.resize(prev_frame, (0,0), None, 0.5, 0.5)

flow_DualTVL1 = cv2.createOptFlow_DualTVL1()

while True:
    status_cap, frame = cap.read()
    frame = cv2.resize(frame, (0,0), None, 0.5,0.5)
    if not status_cap:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not flow_DualTVL1.getUseInitialFlow():
        opt_flow = flow_DualTVL1.calc(prev_frame, gray, None)
        flow_DualTVL1.setUseInitialFlow(True)
    else:
        opt_flow = flow_DualTVL1.calc(prev_frame, gray, opt_flow)

    prev_frame = np.copy(gray)

    if display_flow(frame, opt_flow):
        break

cv2.destroyAllWindows()
'''

#Panorama image using many images
''''''
images = []
images.append(cv2.imread('D:/Depot/Depot_GraduateSchool/IndustrialVision/PanoramaTest01.jpg', cv2.IMREAD_COLOR))
images.append(cv2.imread('D:/Depot/Depot_GraduateSchool/IndustrialVision/PanoramaTest02.jpg', cv2.IMREAD_COLOR))
images.append(cv2.imread('D:/Depot/Depot_GraduateSchool/IndustrialVision/PanoramaTest03.jpg', cv2.IMREAD_COLOR))
images.append(cv2.imread('D:/Depot/Depot_GraduateSchool/IndustrialVision/PanoramaTest04.jpg', cv2.IMREAD_COLOR))

stitcher = cv2.createStitcher()
ret, pano = stitcher.stitch(images)

if ret == cv2.STITCHER_OK:
    pano = cv2.resize(pano, dsize=(0,0), fx=0.8, fy=0.8)
    cv2.imshow('panorama', pano)
    cv2.waitKey()

    cv2.destroyAllWindows()

else:
    print('error during stitching')

