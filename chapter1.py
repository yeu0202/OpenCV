import cv2
import numpy as np
import time
import math
import imutils
print("Package Imported")


cap = cv2.VideoCapture(0)

haloOverlay = cv2.imread('resources/halo.png', cv2.IMREAD_UNCHANGED)
haloHeight, haloWidth, haloChannels = haloOverlay.shape
haloHeight = int(haloHeight/6)
haloWidth = int(haloWidth/6)
haloHalfWidth = int(haloWidth/2)
haloHalfHeight = int(haloHeight/2)

# print(haloHalfWidth)
# print(haloHalfHeight)
haloOverlay = cv2.resize(haloOverlay, (haloWidth, haloHeight))
# cv2.imshow('halo', haloOverlay)


# face detection cascade classifier
faceCascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")

# time calculation
start_time = time.time()
fpsDisplayTime = 1  # displays the frame rate every 1 second
fpsCounter = 0


def overlay_transparent(background, overlay, x, y):
    # print(background.shape)
    # print(overlay.shape)

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
            ],
            axis=2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

# def empty(a):
#     pass
#
# cv2.namedWindow("TrackBars", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("TrackBars", 720, 360)
# cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
# cv2.createTrackbar("Hue Max", "TrackBars", 25, 179, empty)
# cv2.createTrackbar("Sat Min", "TrackBars", 45, 255, empty)
# cv2.createTrackbar("Sat Max", "TrackBars", 163, 255, empty)
# cv2.createTrackbar("Val Min", "TrackBars", 89, 255, empty)
# cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

"""
def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)
"""
#change_res(640, 480);


# font variables
font = cv2.FONT_HERSHEY_SIMPLEX
fontOrigin = (10, 35)
fontScale = 1
fontColour = (255, 255, 255)
fontBorderColour = (0, 0, 0)
fontThickness = 2

def myPutText(img, text):
    outputImage = cv2.putText(img, text, fontOrigin, font, fontScale, fontBorderColour, fontThickness+3, cv2.LINE_AA)
    outputImage = cv2.putText(outputImage, text, fontOrigin, font, fontScale, fontColour, fontThickness, cv2.LINE_AA)
    return outputImage


# stack images for display
def myStackImages(scale, imgArray):
    outputImage = None
    for i in range(len(imgArray)):
        rowStack = np.hstack(imgArray[i])
        if outputImage is not None:
            outputImage = np.vstack((outputImage, rowStack))
        else:
            outputImage = rowStack

    outputImage = cv2.resize(outputImage, (round(outputImage.shape[1]*scale), round(outputImage.shape[0]*scale)))
    return outputImage


# get contours for object detection
# def getContours(img):
#     contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area>200:
#             cv2.drawContours(imgContour, cnt, -1, (255, 255, 0), 3)
#             peri = cv2.arcLength(cnt, True)  # perimeter
#             approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
#             objCorn = len(approx)  # object corners, number of corners the object has
#
#             # draw bounding box
#             x, y, w, h = cv2.boundingRect(approx)
#             cv2.rectangle(imgContour, (x,y), (x+w, y+h), (0,255,255), 2)


def findLargestFace(facesArray):
    index = 0
    for i in range(len(facesArray)-1):
        if facesArray[i+1][2] > facesArray[i][2]:
            index = i+1
    return index


cannyThreshold = 80
cannyRatio = 1.5

imgWidth = 640  # 640
imgHeight = 360  # 360
imgHalfHeight = int(imgHeight/2)
imgHalfWidth = int(imgWidth/2)

frontFace = [0, 0, 0, 0]
previousFaces = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
previousFacesIndex = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # kernel = np.ones((3, 3), np.uint8)  # define a 3 by 3 kernel of ones
    imgResize = cv2.resize(frame, (imgWidth, imgHeight))


    #frameGray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
    #frameBlur = cv2.GaussianBlur(imgResize, (3,3), 0); # (7, 7) is kernel size
    # frameBlur = cv2.blur(imgResize, (1, 1))  # (7, 7) is kernel size
    # frameBlurCanny = cv2.blur(imgResize, (3, 3))  # blur before canny to remove noise
    # imgCanny = cv2.Canny(frameBlurCanny, cannyThreshold, cannyThreshold*cannyRatio)
    #imgDilation = cv2.dilate(imgCanny, kernel, iterations = 1)
    #imgEroded = cv2.erode(imgDilation, kernel, iterations = 1)
    imgGBlurCanny = cv2.GaussianBlur(imgResize, (5, 5), 1)  # Gaussian Blur slightly better for Canny detection
    imgCanny_2 = cv2.Canny(imgGBlurCanny, cannyThreshold, cannyThreshold*cannyRatio)
    #imgDilation = cv2.dilate(imgCanny_2, kernel, iterations = 1)

    # imgHSV = cv2.cvtColor(imgResize, cv2.COLOR_BGR2HSV)
    # h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    # h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    # s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    # s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    # v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    # v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    # lower = np.array([h_min, s_min, v_min])
    # upper = np.array([h_max, s_max, v_max])
    # mask = cv2.inRange(imgHSV, lower, upper)
    # maskedImage = cv2.bitwise_and(imgResize, imgResize, mask=mask)

    # imgContour = imgResize.copy()
    # getContours(imgCanny_2)

    imgFaces = imgResize.copy()
    imgGray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
    cv2.line(imgFaces, (0, imgHalfHeight), (imgWidth, imgHalfHeight), (255, 0, 255), 1)
    cv2.line(imgFaces, (imgHalfWidth, 0), (imgHalfWidth, imgHeight), (255, 0, 255), 1)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(imgFaces, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.line(imgFaces, (x, int(y+(h/2))), (x+w, int(y+(h/2))), (0, 255, 0), 1)
        cv2.line(imgFaces, (int(x+(w/2)), y), (int(x+(w/2)), y+h), (0, 255, 0), 1)


    if len(faces) != 0:
        facesIndex = findLargestFace(faces)
        frontFace = faces[facesIndex]
        # print(faces[facesIndex])
    currentFace = np.mean(np.array([frontFace, previousFaces[0], previousFaces[1], previousFaces[2]]), axis=0)  # smoothing
    previousFaces[previousFacesIndex] = frontFace
    previousFacesIndex += 1
    previousFacesIndex = previousFacesIndex % 3

    faceShiftX = int(currentFace[0] + currentFace[2]/2)
    faceShiftY = int(currentFace[1] + currentFace[3]/2)

    imgWrap = cv2.copyMakeBorder(imgResize, imgHalfHeight, imgHalfHeight, imgHalfWidth, imgHalfWidth, cv2.BORDER_REFLECT_101)
    imgWrap = imgWrap[faceShiftY:faceShiftY + imgHeight,  faceShiftX:faceShiftX + imgWidth]  # array x y are swapped

    imgWrap2 = cv2.copyMakeBorder(imgResize, imgHalfHeight, imgHalfHeight, imgHalfWidth, imgHalfWidth, cv2.BORDER_REFLECT_101)
    imgWrap2 = imgWrap2[(imgHeight - faceShiftY):(imgHeight - faceShiftY) + imgHeight, (imgWidth - faceShiftX):(imgWidth - faceShiftX) + imgWidth]  # array x y are swapped



    # currentFace = frontFace
    #
    # faceShiftX_0 = int(currentFace[0] + currentFace[2]/2)
    # faceShiftY_0 = int(currentFace[1] + currentFace[3]/2)
    #
    # imgWrap_0 = cv2.copyMakeBorder(imgResize, imgHalfHeight, imgHalfHeight, imgHalfWidth, imgHalfWidth, cv2.BORDER_REFLECT_101)
    # imgWrap_0 = imgWrap_0[faceShiftY_0:faceShiftY_0 + imgHeight,  faceShiftX_0:faceShiftX_0 + imgWidth]  # array x y are swapped
    #
    # imgWrap2_0 = cv2.copyMakeBorder(imgResize, imgHalfHeight, imgHalfHeight, imgHalfWidth, imgHalfWidth, cv2.BORDER_REFLECT_101)
    # imgWrap2_0 = imgWrap2_0[(imgHeight - faceShiftY_0):(imgHeight - faceShiftY_0) + imgHeight, (imgWidth - faceShiftX_0):(imgWidth - faceShiftX_0) + imgWidth]  # array x y are swapped



    blurIndex = math.ceil(abs((faceShiftX - imgHalfWidth))/(imgHalfWidth/30))
    blurIndex = abs(blurIndex)
    blurIndex = blurIndex * 2 - 1
    blurIndex = abs(blurIndex)
    # print(blurIndex)
    # imgBlur = cv2.blur(imgResize, (blurIndex, blurIndex))  # (7, 7) is kernel size

    imgMovingBlur = cv2.blur(imgWrap2, (blurIndex, blurIndex))

    # imgRotate = cv2.copyMakeBorder(imgResize, imgWidth, imgWidth, imgWidth, imgWidth, cv2.BORDER_REFLECT_101)
    # imgRotate = imutils.rotate_bound(imgRotate, 30)
    # imgRotateHalfHeight = int(imgRotate.shape[0]/2)
    # imgRotateHalfWidth = int(imgRotate.shape[1]/2)
    # imgRotate = imgRotate[imgRotateHalfHeight - imgHalfHeight:imgRotateHalfHeight + imgHalfHeight, imgRotateHalfWidth - imgHalfWidth: imgRotateHalfWidth + imgHalfWidth]
    #
    # imgRotate2 = cv2.copyMakeBorder(imgResize, imgWidth, imgWidth, imgWidth, imgWidth, cv2.BORDER_REFLECT_101)
    # imgRotate2 = imutils.rotate_bound(imgRotate2, abs(faceShiftY - imgHalfHeight)/(imgHalfWidth/1440))
    # imgRotateHalfHeight = int(imgRotate2.shape[0] / 2)
    # imgRotateHalfWidth = int(imgRotate2.shape[1] / 2)
    # imgRotate2 = imgRotate2[imgRotateHalfHeight - imgHalfHeight:imgRotateHalfHeight + imgHalfHeight, imgRotateHalfWidth - imgHalfWidth: imgRotateHalfWidth + imgHalfWidth]


    # Convert grayscale to BGR
    imgCannyBGR = cv2.cvtColor(imgCanny_2, cv2.COLOR_GRAY2BGR)
    #imgDilation3 = cv2.cvtColor(imgDilation, cv2.COLOR_GRAY2BGR)
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # imgCannyBGR = cv2.bitwise_not(imgCannyBGR)
    # imgCannyCombine2 = cv2.bitwise_and(imgResize, imgCannyBGR)
    imgWhite = np.full(imgResize.shape, 255, dtype=np.uint8)
    imgForeground = cv2.bitwise_or(imgWhite, imgWhite, mask=imgCanny_2)
    imgCanny_2 = cv2.bitwise_not(imgCanny_2)
    imgBackground = cv2.bitwise_or(imgResize, imgResize, mask=imgCanny_2)
    imgCannyCombine = cv2.bitwise_or(imgForeground, imgBackground)

    alpha = abs((faceShiftX - imgHalfWidth))/(imgHalfWidth/2)
    alpha = np.clip(alpha, 0, 1)
    imgCannyCombine = cv2.addWeighted(imgCannyCombine, alpha, imgResize, 1-alpha, 0)


    b, g, r = cv2.split(imgCannyCombine)
    b2, g2, r2 = cv2.split(imgMovingBlur)
    b3, g3, r3 = cv2.split(imgWrap)
    imgCombined = cv2.merge((b, g2, r3))


    imgWrapGaussian = cv2.GaussianBlur(imgWrap, (5, 5), 1)  # Gaussian Blur slightly better for Canny detection
    imgWrapCanny = cv2.Canny(imgWrapGaussian, cannyThreshold, cannyThreshold*cannyRatio)

    imgWhite = np.full(imgResize.shape, 255, dtype=np.uint8)
    imgForeground = cv2.bitwise_or(imgWhite, imgWhite, mask=imgWrapCanny)
    imgWrapCanny = cv2.bitwise_not(imgWrapCanny)
    imgBackground = cv2.bitwise_or(imgWrap, imgWrap, mask=imgWrapCanny)
    imgCannyCombine2 = cv2.bitwise_or(imgForeground, imgBackground)
    imgCannyCombine2 = cv2.addWeighted(imgCannyCombine2, alpha, imgWrap, 1-alpha, 0)

    imgWrapBlur = cv2.blur(imgWrap, (blurIndex, blurIndex))

    imgExample = imgResize.copy()
    imgExampleCanny = cv2.cvtColor(imgCanny_2, cv2.COLOR_GRAY2BGR)
    imgExample = cv2.bitwise_and(imgExample, imgExampleCanny)

    b, g, r = cv2.split(imgCannyCombine2)
    b2, g2, r2 = cv2.split(imgWrapBlur)
    b3, g3, r3 = cv2.split(imgWrap)
    imgCombined2 = cv2.merge((b, g2, r3))


    alpha = abs((faceShiftX - imgHalfWidth))/(imgHalfWidth/2)
    alpha = np.clip(alpha, 0, 1)
    imgCannyCombine = cv2.addWeighted(imgCannyCombine, alpha, imgResize, 1-alpha, 0)


    # Add text to images
    imgResize = myPutText(imgResize, 'Camera')
    # imgCannyBGR = myPutText(imgCannyBGR, 'Canny')
    #imgGBlurCanny = myPutText(imgGBlurCanny, 'Blur')
    #imgDilation3 = myPutText(imgDilation3, 'Thick Canny')
    # mask = myPutText(mask, 'Mask')
    # maskedImage = myPutText(maskedImage, 'Masked Image')
    # imgHSV = myPutText(imgHSV, 'HSV')
    # imgContour = myPutText(imgContour, 'Object detection')
    imgFaces = myPutText(imgFaces, 'Face detection')
    # imgWrap = myPutText(imgWrap, 'Face following')
    # imgWrap2 = myPutText(imgWrap2, 'Face dependent offset')
    # imgBlur = myPutText(imgBlur, 'Face dependent blur')
    # imgMovingBlur = myPutText(imgMovingBlur, 'Face dependent blur+offset')
    # imgRotate = myPutText(imgRotate, 'Rotation')
    # imgRotate2 = myPutText(imgRotate2, 'Face dependent rotation')
    # imgCannyCombine = myPutText(imgCannyCombine, 'Face dependent Canny')

    # Stack the images together
    #imgVer = np.vstack((img, img))
    #outputFrame = np.hstack((imgResize, imgCanny3))
    outputFrame = myStackImages(1, [[imgResize, imgFaces], [imgCombined, imgCombined2]])
    # resultsFrame = myStackImages(1, [[imgCombined, imgCombined2]])

    """
    imgAngel = imgResize.copy()
    # tempOverlay = cv2.addWeighted(imgAngel[250:250 + rows, 0:0 + cols], 0.5, haloOverlay, 0.5, 0)
    # imgAngel[250:250 + rows, 0:0 + cols] = tempOverlay
    imgAngel = overlay_transparent(imgAngel, haloOverlay, int(currentFace[0]), 100)
    """

    # Display the resulting frame
    #cv2.imshow('Video1', frame)
    cv2.imshow('Video', outputFrame)
    # cv2.imshow('video2', imgWhite)
    # cv2.imshow('video3', imgForeground)
    # cv2.imshow('video', imgAngel)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    fpsCounter += 1
    if (time.time() - start_time) > fpsDisplayTime:
        print("FPS: ", fpsCounter / (time.time() - start_time))
        fpsCounter = 0
        start_time = time.time()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()







""" Shows an image file
img = cv2.imread("Resources/testImage.png");

cv2.imshow("Output", img);
cv2.waitKey(0);
"""

""" Displays a video, can quit early with pressing the key 'q'
cap = cv2.VideoCapture("Resources/testVideo.MP4");
while True:
    success, img = cap.read(); #success is bool to store wether read was successful
    cv2.imshow("Video", img);
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
"""

""" Displays camera feed, press 'q' to quit
cap = cv2.VideoCapture(0);

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('Video',frame);
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
"""