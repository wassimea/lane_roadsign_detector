import cv2
import numpy as np
import math
from os import listdir
from os.path import isfile, join
import os
def region(img,vertices):
	mask = np.zeros_like(img)
	match_mask_color=  255   
	cv2.fillPoly(mask,vertices,match_mask_color)
	masked_image=cv2.bitwise_and(img,mask)
	return masked_image

def draw_the_lines(img,lines):
    imge=np.copy(img)
    blank_image=np.zeros((imge.shape[0],imge.shape[1],3),dtype=np.uint8)

    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope  = abs((y2 - y1) / (x2 - x1))
                if slope > 0.30:
                    cv2.line(blank_image,(x1,y1),(x2,y2),(0,255,0),thickness=3)

        imge = cv2.addWeighted(imge,0.8,blank_image,1,0.0)
    return imge

def check_angles(shape, img):
    x = 1
    p1, p2, p3, p4 = shape
    p1x, p1y = p1[0]
    p2x, p2y = p2[0]
    p3x, p3y = p3[0]
    p4x, p4y = p4[0]

    a1r = math.atan2(p4y - p1y, p4x - p1x) - math.atan2(p2y - p1y, p2x - p1x)
    a2r = math.atan2(p3y - p4y, p3x - p4x) - math.atan2(p1y - p4y, p1x - p4x)
    a3r = math.atan2(p2y - p3y, p2x - p3x) - math.atan2(p4y - p3y, p4x - p3x)
    a4r = math.atan2(p1y - p2y, p1x - p2x) - math.atan2(p3y - p2y, p3x - p2x)


    a1d = math.degrees(a1r)
    if a1d < 0 :
        a1d = 180 - abs(a1d)
    a2d = math.degrees(a2r)
    if a2d < 0 :
        a2d = 180 - abs(a2d)
    a3d = math.degrees(a3r)
    if a3d < 0 :
        a3d = 180 - abs(a3d)
    a4d = math.degrees(a4r)
    if a4d < 0 :
        a4d = 180 - abs(a4d)
    
    if abs(90 - abs(a1d)) > 15:
        return False
    if abs(90 - abs(a2d)) > 15:
        return False
    if abs(90 - abs(a3d)) > 15:
        return False
    if abs(90 - abs(a4d)) > 15:
        return False

    return True

def get_mser(mser, image):
    img = image.copy()
    height, width, _ = img.shape
    roi = img[0:700, int(width/2):]

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    rects = []
    regions = mser.detectRegions(roi)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
    for hull in hulls:
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.005*peri, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            if w >50 and h > 50:
                if check_angles(approx, roi):
                
                    rects.append([int(width/2) + x,y,int(width/2) + x+w,y+h])
    return rects


def process_images(folder_path):
    #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    #out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1920,1080))

    image_filenames = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    mser = cv2.MSER_create()

    index = 0
    while os.path.exists(folder_path + str(index) + ".png"):
        frame = cv2.imread(folder_path + str(index) + ".png")
        cop = frame.copy()
        height, width, _ = frame.shape
        rects = get_mser(mser, frame)

        low_left = (200,height - 400)

        mid = (width/2, height/3 + 70)
        bottom_right = (width- 350,height - 400)

        region_of_interest_coor = [ low_left,mid,bottom_right]

        edges = cv2.Canny(frame,80,200)
        isolated = region(edges, np.array([region_of_interest_coor],np.int32))
        lines = cv2.HoughLinesP(isolated, 1, np.pi/180, 40, np.array([]), minLineLength=50, maxLineGap=50)


        image_with_lines = draw_the_lines(frame,lines)

        for r in rects:
            x1 = r[0]
            y1 = r[1]
            x2 = r[2]
            y2 = r[3]
            cv2.rectangle(image_with_lines, (x1, y1), (x2, y2), (0,0,255), 2)
        cv2.imshow("image_with_lines", image_with_lines)
        cv2.waitKey(1)
        index += 1

        #out.write(image_with_lines)
    return


def main():
    process_images("/media/wassimea/Storage/invest_ottawa/lane-detector/images/")



if __name__ == "__main__":
    main()

