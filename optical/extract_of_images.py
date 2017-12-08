import __init__

base_path = __init__.base_path
import os,sys,cv2
from glob import glob
import numpy as np

from video import get_frames

videosPath = os.path.join(base_path, '..','code','trailers')
framesPath = os.path.join(base_path, 'frames')
###
outPath = os.path.join(base_path,'flow')

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    #print np.unique(cv2.normalize(fx,0,255,cv2.NORM_L1).astype('uint8'))
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def bgr2gray(frame):
    return np.mean(frame,axis=-1)

def calc_optical_flow(ID):

    assert type(ID) == str, "Print Please pass ID to calc_optical_flow"

    videoPath = glob(os.path.join(videosPath,'%s.mp4'%ID))
    if videoPath == []:
        videoPath = glob(os.path.join(videosPath,'%s.webm'%ID))
    videoPath = videoPath[0]

    flowPath = os.path.join(outPath,ID)

    flowPath = os.path.join(outPath,ID)
    if not os.path.exists(flowPath):
        os.makedirs(flowPath)

    cap = cv2.VideoCapture(videoPath)

    ret, frame1 = cap.read()
    frame1 = cv2.resize(frame1, (224,224))
    prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    count = 0
    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        count +=1 
        if count % 5 == 0:
            #picking every fifth frame

            frame2= cv2.resize(frame2, (224,224))
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev, next, 0.5,3,15,3,5,1.2,0)
            horz = cv2.normalize(flow[...,0],  0, 255, cv2.NORM_MINMAX)
            vert = cv2.normalize(flow[...,1],  0, 255, cv2.NORM_MINMAX)
            horz = horz.astype('uint8')
            vert = vert.astype('uint8')

            cv2.imwrite(os.path.join(flowPath,'flow_%03d_h.jpg' % count), cv2.cvtColor(horz,cv2.COLOR_GRAY2BGR),[int(cv2.IMWRITE_JPEG_QUALITY), 90])
            cv2.imwrite(os.path.join(flowPath,'flow_%03d_v.jpg' % count), cv2.cvtColor(vert,cv2.COLOR_GRAY2BGR),[int(cv2.IMWRITE_JPEG_QUALITY), 90])
            cv2.imwrite(os.path.join(flowPath,'repr_%03d.jpg' % count),
                    np.hstack((bgr2gray(draw_flow(prev,flow)),cv2.cvtColor(draw_hsv(flow), cv2.COLOR_BGR2GRAY),horz,vert)))

            prev = next
calc_optical_flow('3809')
