# Developed by Lateraluz++
# 24-01-2022 
# Get streams from vigilance WebCam and detect faces.

import cv2 as cv
#import numpy as npimport 
#import sys as os



def get_stream_from_webcam():

    RTSP_URL = "rtsp://192.168.0.20/live/ch00_2"     


    cascate_file_path = 'haarcascade_frontalface_default.xml'
    faceCascade = cv.CascadeClassifier(cascate_file_path)

    video_capture = cv.VideoCapture(RTSP_URL)
    flag = True
    while(flag):        
        ret, frame = video_capture.read()       

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
                                            gray,                     # image gray 
                                            scaleFactor  = 1.1,       # specifying how much the image size is reduced at each image scale.
                                            minNeighbors = 5,         # specifying how many neighbors each candidate rectangle should have to retain it.
                                            minSize      = (42, 42),   # Minimum possible object size. Objects smaller than that are ignored
                                            flags        = cv.CASCADE_SCALE_IMAGE  # Fixed with new version, replace flags=cv2.CV_HAAR_SCALE_IMAGE
                                            )

        # Draw a rectangle around the faces detected
        for (x, y, w, h) in faces:        
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Resize  UI Window output
        frameResized = cv.resize(frame, (960, 540)) 
        cv.imshow(RTSP_URL, frameResized)
        #cv.waitKey(1)

        key = cv.waitKey(1)
        # close window if keystroke is either q or Q
        if key == ord('q') or key == ord('Q'):
            flag = False
        else:
            flag = True
            
    # Close window and Release resources. It is mandatory to avoid freeze the UI 
    video_capture.release()
    cv.destroyAllWindows()            

    
def main():    
    get_stream_from_webcam()
          
    
if __name__ == "__main__":
    main()
    


