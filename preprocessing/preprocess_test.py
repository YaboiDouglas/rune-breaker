import cv2
import numpy as np
import common
import colorful as cf
from matplotlib import pyplot as plt

SEARCH_REGION_WIDTH = 450


def main():
    common.create_directories()
    labeled_imgs = common.get_files(common.LABELED_DIR)
    approved = 0
    count = 0
    for path, filename in labeled_imgs:
        #print("Processing " + cf.skyBlue(path))
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
        # Apply Gaussian blur to reduce noise
        img_blur = cv2.GaussianBlur(RGB_img, (3, 3), 0)
  
        # Apply Canny edge detection
        edges = cv2.Canny(img_blur, 50, 150)

        # Apply Hough Circle Transform to detect circles
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=25, minRadius=18, maxRadius=33)

        print(get_circle_centers(circles,img))
        if count > 20:
            exit()
        count += 1
        

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

if __name__ == "__main__":
    main()