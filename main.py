import cv2 as cv

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    success, frame = cap.read()

    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # vision (for object detection we dont need color we use gray we care about brightness)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #convert image to black and white  #it returns 2 output the first one  is the the threshhold that used so we want second one which is real output
    _,threshhold = cv.threshold(gray,127,255,cv.THRESH_BINARY)
    #threshhold_adaptive = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2) TODO: it gives more info but its hard to put harness around it! find a way to implement it

    #find object contours
    # RETR_TREE will get contours at multiple hierarchy levels including the ones inside others!
    # we want outer boundry of chuncks so I changed it to RETR_EXTERNAL
    contours,hierarchy = cv.findContours(threshhold,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    # Select the largest contour
    if contours:
        chunck = max(contours, key = cv.contourArea)
        area = cv.contourArea(chunck)
        # ignor small obj
        if area > 500:
            x,y,w,h = cv.boundingRect(chunck)
            
            cv.drawContours(frame,[chunck],-1,(0,0,255),2)

            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            cv.putText(frame,f"Area: {area:.0f}px",(x,y-10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

            cv.imshow("Object Detection",frame)
            cv.imshow("Threshold",threshhold)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break



    #draw rectangle around detected object
    '''
     for con in contours:
        x,y,w,h = cv.boundingRect(con)

        #ignore very small noise
        if w * h > 500:
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv.putText(frame,f"W: {w}px H: {h}px",(x,y -10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    
    '''

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()