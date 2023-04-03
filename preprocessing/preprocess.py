import argparse
import os
import re
import uuid
import sys

sys.path.insert(0, os.path.abspath('.'))

import colorful as cf
import cv2
import numpy as np
import pandas as pd
from skimage import morphology
from matplotlib import pyplot as plt

import common

OUTPUT_WIDTH = common.INPUT_SHAPE[0]

ARROW_BOX_DIST = 100
SEARCH_REGION_WIDTH = 450
SEARCH_REGION_HEIGHT = 100
HALF_ARROW_WIDTH = 35
HALF_ARROW_HEIGHT = 35

EXIT_KEY = 113 # q
APPROVE_KEY = 32 # space


def main(inspection, mode, automatic):
    common.create_directories()

    print("     SPACE = approve")
    print("OTHER KEYS = skip")
    print("         Q = quit\n")

    labeled_imgs = common.get_files(common.LABELED_DIR)

    approved = 0
    for path, filename in labeled_imgs:
        print("Processing " + cf.skyBlue(path))

        # Load the image
        display = cv2.imread(path)
        height, width, _ = display.shape
        arrows = []
        # manually tuned values
        search_x, search_y = width // 5 + 15, height // 4
        search_width, search_height = SEARCH_REGION_WIDTH, height // 2 - search_y

        x0 = search_x
        x1 = x0 + search_width

        y0 = search_y
        y1 = y0 + search_height

        img = display[y0:y1, x0:x1]

        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if RGB_img is None:
            print('Processing failed for: '+cf.skyBlue(path))
            pass
        # Apply Gaussian blur to reduce noise
        img_blur = cv2.GaussianBlur(RGB_img, (3, 3), 0)
  
        # Apply Canny edge detection
        edges = cv2.Canny(img_blur, 50, 150)

        # Apply Hough Circle Transform to detect circles
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=25, minRadius=18, maxRadius=33)

        circle_coords = get_circle_centers(circles,img)
        
        if circle_coords == []:
            print("No elements in array, arrow's not found.")
            cv2.imshow('failed',RGB_img)
            cv2.waitKey()
            break
        circle_coords.sort()

        for x,y in circle_coords:
            x0 = x - HALF_ARROW_WIDTH
            x1 = x + HALF_ARROW_WIDTH

            y0 = y - HALF_ARROW_HEIGHT
            y1 = y + HALF_ARROW_HEIGHT

            img = RGB_img[y0:y1, x0:x1]
            #cv2.imshow('cropped original arrow',img)
            #cv2.waitKey()
            #(cx, cy), arrow_box = process_arrow(img, mode)

            #search_x += int(cx + ARROW_BOX_DIST - SEARCH_REGION_WIDTH / 2)
            #search_y += int(cy - SEARCH_REGION_HEIGHT / 2)

            #search_width = SEARCH_REGION_WIDTH
            #search_height = SEARCH_REGION_HEIGHT
            processed_arrow = process_arrow(img,'binarized')
            arrows.append(processed_arrow)

        if not automatic:
            arrow_type, directions, _ = re.split('_', filename)
            reference = get_reference_arrows(directions, arrows[0].shape)

            cv2.imshow(arrow_type, np.vstack([np.hstack(arrows), reference]))

            key = cv2.waitKey()
            cv2.destroyAllWindows()
        else:
            key = APPROVE_KEY

        if key == APPROVE_KEY:
            if not inspection:
                save_arrow_imgs(arrows, filename)
            approved += 1
        elif key == EXIT_KEY:
            break
        else:
            print("Skipped!")

    if len(labeled_imgs) > 0:
        print("\nApproved {} out of {} images ({}%).\n".format(
            approved, len(labeled_imgs), 100 * approved // len(labeled_imgs)))
    else:
        print("There are no images to preprocess.\n")

    show_summary()

    print("Finished!")


def process_arrow(img, mode):
    # gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    coefficients = (0.0445, 0.6568, 0.2987) # (h, s, v)
    img = cv2.transform(img, np.array(coefficients).reshape((1, 3)))
    if mode == 'gray':
        output = img.copy()
    # binarization
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5,-1)
    # noise removal
    denoise(img, threshold=8, conn=2)
    if mode == 'binarized':
        output = img.copy()
    '''
    #we don't need this because of the other arrow location I did for ellinia arrows
    # processing
    cx, cy = compute_arrow_centroid(img)

    # result cropping
    max_height, max_width = img.shape

    x0 = max(int(cx - OUTPUT_WIDTH / 2), 0)
    y0 = max(int(cy - OUTPUT_WIDTH / 2), 0)

    x1 = int(x0 + OUTPUT_WIDTH)
    if x1 >= max_width:
        x0 -= x1 - max_width
        x1 = max_width

    y1 = int(y0 + OUTPUT_WIDTH)
    if y1 >= max_height:
        y0 -= y1 - max_height
        y1 = max_height

    box = output[y0:y1, x0:x1]
    
    return (cx, cy), box
    '''
    return output

def denoise(img, threshold=64, conn=2):
    processed = img > 0

    processed = morphology.remove_small_objects(
        processed, min_size=threshold, connectivity=conn)

    processed = morphology.remove_small_holes(
        processed, area_threshold=threshold, connectivity=conn)

    mask_x, mask_y = np.where(processed == True)
    img[mask_x, mask_y] = 255

    mask_x, mask_y = np.where(processed == False)
    img[mask_x, mask_y] = 0

def get_circle_centers(circles,img):
    circle_count = 0
    if circles is not None:
            x_list = []
            final_list = []
            # Convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Extract a circular ROI
                roi = img[y-r:y+r, x-r:x+r]

                # Calculate the mean color of the ROI
                mean_color = cv2.mean(roi)

                # Check if the mean color of the ROI is multi-colored
                if len(mean_color) > 3 and (18<r<33):
                    x_list.append(x)
                    circle_count+=1
                    #print(f"Multi-colored circle found at ({x}, {y}) with radius {r}")
                    #cv2.circle(RGB_img,center_coords,r,(0,0,255),1)
                    final_list.append([x,y])
            #sort if there are more than 4 centers found
            if((circle_count > 4) and (not check_interval(x_list))):
                #if we need sort we replace final list completely by averaging values where possible
                final_list = average_close_numbers(circles)
                return final_list
            else:
                if(circle_count == 4):
                    #return xy list
                    return final_list
                #otherwise we couldn't sort and fix issue or there were too little so we just say
                #fuck it and re-do the rune by returning a blank list
                if not (circle_count == 4):
                    return []
                
def check_interval(lst):
    lst.sort()  # Sort the list in ascending order
    for i in range(len(lst)-1):
        if lst[i+1] - lst[i] < 60:
            return False
    return True

def average_close_numbers(lst):
    b_just_sorted = False
    #sort list on x element (index 0)
    lst = lst[np.argsort(lst[:,0])]
    #result output
    result = []
    xy_result = []
    i = 0
    #while there is still stuff in the list, keep iterating
    while i < len(lst):
        #we are at the end so exit loop
        if i == len(lst) - 1:
            #if we are at the end and we just sorted that means we have nothing left, otherwise we need to add the last element
            if not b_just_sorted:
                result.append(lst[i])
            break
        #check x values and if less than 60 differece then we need to add the averages to the result set
        if lst[i+1][0] - lst[i][0] < 60:
            avg_x = int(sum([lst[i][0], lst[i+1][0]]) / 2)
            avg_y = int(sum([lst[i][1], lst[i+1][1]]) / 2)
            avg_r = int(sum([lst[i][2], lst[i+1][2]]) / 2)
            
            result.append([avg_x, avg_y, avg_r])
            #delete array rows so we don't go through them again
            lst = np.delete(lst,i+1,0)
            lst = np.delete(lst,i,0)
            b_just_sorted = True
        #otherwise business as usual
        else:
            b_just_sorted = False
            result.append(lst[i])
            i += 1
    #convert to x,y list and return
    for (x, y, r) in result:
        xy_result.append([x,y])
    return xy_result

def compute_arrow_centroid(img):
    contours = cv2.findContours(
        img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # filter contours by area
    candidates = []

    for contour in contours:
        score, (cx, cy), area = circle_features(contour)
        if area > 748 and area < 3600:
            #print('Area: '+str(area))
            candidates.append(((cx, cy), score))

    if candidates:
        match = max(candidates, key=lambda x: x[1])
        (cx, cy), score = match
        if score > 0.5:
            return (int(cx), int(cy))

    print("Centroid not found! Returning the center point...")
    
    height, width = img.shape
    return (width // 2, height // 2)


def circle_features(contour):
    hull = cv2.convexHull(contour)
    
    if len(hull) < 5:
        return 0, (-1, -1), -1

    hull_area = cv2.contourArea(hull)

    (ex, ey), (d1, d2), angle = cv2.fitEllipse(hull)
    ellipse_area = np.pi * (d1 / 2) * (d2 / 2)

    (cx, cy), r = cv2.minEnclosingCircle(hull)
    circle_area = np.pi * r ** 2

    s1 = abs(ellipse_area - hull_area) / max(ellipse_area, hull_area)
    s2 = abs(ellipse_area - circle_area) / max(ellipse_area, circle_area)

    score = 1 - np.mean([s1, s2])

    return score, (ex, ey), ellipse_area


def get_reference_arrows(directions, shape):
    reference = []

    for d in directions:
        arrow = np.zeros(shape, dtype=np.uint8)

        w, h = shape[1], shape[0]
        cx, cy = w // 2, h // 3

        # upward arrow
        points = np.array([(cx - w // 5, cy + h // 8),
                           (cx + w // 5, cy + h // 8),
                           (cx, cy - h // 8)])

        cv2.fillConvexPoly(arrow, points, (255, 255, 255))
        cv2.line(arrow, (cx, cy), (cx, 3 * h // 5), (255, 255, 255), 10)

        rotations = 0

        if d == 'r':
            rotations = 1
        elif d == 'd':
            rotations = 2
        elif d == 'l':
            rotations = 3

        for _ in range(rotations):
            arrow = cv2.rotate(arrow, cv2.ROTATE_90_CLOCKWISE)

        reference.append(arrow)

    return np.hstack(reference)


def save_arrow_imgs(arrows, labeled_filename):
    words = re.split('_', labeled_filename)
    arrow_type = words[0]
    directions = words[1]

    # save individual arrows + their rotated and flipped versions
    for x, arrow_img in enumerate(arrows):
        for rotation in range(4):
            if rotation > 0:
                arrow_img = cv2.rotate(arrow_img, cv2.ROTATE_90_CLOCKWISE)

            direction = get_direction(directions[x], rotation)
            arrow_path = "{}{}_{}_{}".format(common.SAMPLES_DIR, arrow_type, direction, uuid.uuid4())

            cv2.imwrite(arrow_path + ".png", arrow_img)

            if direction in ['down', 'up']:
                flipped_img = cv2.flip(arrow_img, 1)
            else:
                flipped_img = cv2.flip(arrow_img, 0)
            
            cv2.imwrite(arrow_path + "F.png", flipped_img)

    os.rename(common.LABELED_DIR + labeled_filename,
              common.PREPROCESSED_DIR + labeled_filename)


def get_direction(direction, rotation):
    direction_dict = {
        'l': 'left',
        'u': 'up',
        'r': 'right',
        'd': 'down'
    }
    rotation_list = ['l', 'u', 'r', 'd']

    new_index = (rotation_list.index(direction) +
                 rotation) % len(rotation_list)
    new_direction = rotation_list[new_index]

    return direction_dict[new_direction]


def show_summary():
    matrix = pd.DataFrame(np.zeros((4, 5), dtype=np.int32), index=(
        'round', 'wide', 'narrow', 'total'), columns=('down', 'left', 'right', 'up', 'total'))

    images = common.get_files(common.SAMPLES_DIR)

    for _, filename in images:
        arrow_direction, arrow_type = common.arrow_labels(filename)

        matrix[arrow_direction][arrow_type] += 1

        matrix['total'][arrow_type] += 1
        matrix[arrow_direction]['total'] += 1
        matrix['total']['total'] += 1

    print(cf.salmon("Samples summary"))
    print(matrix, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inspection', action='store_true',
                        help="Toggles the inspection mode, which disables the output")
    parser.add_argument('-m', '--mode', default='binarized', type=str,
                        choices=['binarized', 'gray'],
                        help="Sets the output mode to binarized or grayscale")
    parser.add_argument('-a', '--automatic', action='store_true',
                        help="Toggles the automatic mode, which approves all screenshots")

    args = parser.parse_args()

    main(args.inspection, args.mode, args.automatic)
