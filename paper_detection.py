import cv2
import imutils
import numpy as np

cap = cv2.VideoCapture(0)
smallImage = cv2.imread("cat.png")


while True:
    _, image = cap.read()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0]
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    screenContour = None
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            screenContour = approx
            moments = cv2.moments(contour)
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            break

    if screenContour is not None:
        cv2.drawContours(image, [screenContour], -1, (0, 255, 0), 5)

        _,_,boundingBoxWidth,boundingBoxHeight = cv2.boundingRect(screenContour)

        smallImage = cv2.resize(smallImage, (50,50))

        smallImageWidth = smallImage.shape[1]
        smallImageHeight = smallImage.shape[0]

        x0 = int(cX - smallImageWidth/2)
        y0 = int(cY - smallImageHeight/2)
        x1 = x0 + smallImageWidth
        y1 = y0 + smallImageHeight

        if (boundingBoxWidth >= 50) and (boundingBoxHeight >= 50):
            image[y0:y1, x0:x1] = smallImage

    cv2.imshow("Image", image)
    cv2.waitKey(1)
