import gdi_capture
import cv2
import time
import math
import keyboard

class ScreenShot:
    def __init__(self):
        self.hwnd = gdi_capture.find_window_from_executable_name("MapleStory.exe")

    def screenCap(self):
        with gdi_capture.CaptureWindow(self.hwnd) as img:
            if img.any():
                cv2.imwrite('data/screenshots/'+str(math.trunc(time.time()))+'.png',img)

if __name__ == "__main__":
    sc = ScreenShot()
    while True:
        if keyboard.read_key() == "l":
            print('taking screenshot')
            sc.screenCap()
        if keyboard.read_key() == "p":
            print("Exiting")
            break
    
