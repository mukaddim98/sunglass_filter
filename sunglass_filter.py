# references:
# https://www.codespeedy.com/depth-map-from-stereo-images-opencv-python/
# https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/
# https://www.kdnuggets.com/2019/08/introduction-image-segmentation-k-means-clustering.html
# https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/


# Image scaling function
def scaleImg(img, percentage):
    scale_percent = percentage # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    scaledImg = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return scaledImg

# Segmentation scaling function
def KMeansSeg(img, K=3, attempts=10):
    vectorized = img.reshape((-1,3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

    center = np.uint8(center)

    res = center[label.flatten()]
    result_image = res.reshape((img.shape))

    return result_image

# Main filter function
def sunglassFilter(imgLPath, imgRPath, sunglassImgPath, background=False):

    ''' CREATING DEPTH MAP '''

    # load left and right images
    imgL = cv2.imread(imgLPath)
    imgR = cv2.imread(imgRPath)
    imgL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
    imgR = imgR[0:imgL.shape[0], 0:imgL.shape[1]]
    imgL = imgL[0:imgR.shape[0], 0:imgR.shape[1]]

    # Display
    plt.imshow(imgL,"gray")
    plt.show()
    plt.imshow(imgR,"gray")
    plt.show()

    # reduce images size
    print("Reducing imgL and imgR to 40%")
    imgR = scaleImg(imgR, 40)
    imgL = scaleImg(imgL, 40)

    # Gaussian blur the images
    print("Blurring imgL and imgR by 5x5 kernel")
    imgR = cv2.GaussianBlur(imgR,(5,5),cv2.BORDER_DEFAULT)
    imgL = cv2.GaussianBlur(imgL,(5,5),cv2.BORDER_DEFAULT)

    # Display
    plt.imshow(imgL,"gray")
    plt.show()
    plt.imshow(imgR,"gray")
    plt.show()

    # create depth map
    print("Creating depth map")
    stereo = cv2.StereoBM_create(numDisparities=240, blockSize=25)
    disparity = stereo.compute(imgL,imgR)

    # Display
    plt.imshow(disparity,"gray")
    plt.show()

    print("Reducing depth map to 10%")
    redPc = 10
    disparity = scaleImg(disparity, redPc)

    # Gaussian blur depth map
    print("Blurring depth map by 51x51 kernel")
    disparity = cv2.GaussianBlur(disparity,(51,51),cv2.BORDER_DEFAULT)

    print("Increasing depth map to 100%")
    disparity = scaleImg(disparity, (10000/redPc) )

    # Cleaning depth map by removing noise
    print("Removing noise in depth map")

    # Threshold selected based on background
    if background:
        threshold = 450
        k = 7
    else:
        threshold = 100
        k = 5

    for pi in range(disparity.shape[0]):
        for pj in range(disparity.shape[1]):
            if disparity[pi][pj] <= threshold: #if less than threshold we set them to constant value 
                disparity[pi][pj] = -15
            else:
                disparity[pi][pj] = threshold #if more than threshold we set them to threshold value

    # Display
    plt.imshow(disparity,"gray")
    plt.show()       
    
    ''' CREATING A WHITE BACKGROUND AND SEGMNETATION '''

    # Convert depth mapth to binary
    print("Converting depth mapth to binary")
    _, disparity = cv2.threshold(disparity, 0, 255, cv2.THRESH_BINARY_INV)

    # Display
    plt.imshow(disparity,"gray")
    plt.show()

    print("Creating white background")
    whiteBG = imgL.copy()
    cv2.imwrite('whiteBG.jpg', whiteBG)
    whiteBG = cv2.imread('whiteBG.jpg')
    os.remove('whiteBG.jpg')

    for pi in range(disparity.shape[0]):
        for pj in range(disparity.shape[1]):
            if disparity[pi][pj] == 255:
                whiteBG[pi][pj] = 255

    whiteBG = scaleImg(whiteBG, 40)
    whiteBG = scaleImg(whiteBG, 250)
    whiteBG = cv2.GaussianBlur(whiteBG,(5,5),cv2.BORDER_DEFAULT)

    # Display
    plt.imshow(whiteBG,"gray")
    plt.show()

    print("Segmenting the image")
    segmented = KMeansSeg(whiteBG, K=k)

    # Display
    plt.imshow(segmented,"gray")
    plt.show()

    ''' PLACING THE SUNGLASS '''

    # Load two images
    face = cv2.imread(imgLPath)
    sunglass = cv2.imread(sunglassImgPath)
    
    # Locate the two eyes
    print("Locating the eyes")
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

    cv2.imwrite('segmented.jpg', segmented)
    imgEye = cv2.imread('segmented.jpg')
    os.remove('segmented.jpg')
    grayEye = cv2.cvtColor(imgEye, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayEye, 1.3, 5)
    eyeCenters = []

    for (x,y,w,h) in faces:
        cv2.rectangle(imgEye,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = grayEye[y:y+h, x:x+w]
        roi_color = imgEye[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eyeCenters.append([(2*x+2*ex+ew)//2, (2*y+2*ey+eh)//2])

    if(eyeCenters[0][0] > eyeCenters[1][0]):
        eyeCenters[0][0], eyeCenters[1][0] = eyeCenters[1][0], eyeCenters[0][0]

    # Display
    plt.imshow(imgEye,"gray")
    plt.show()

    # Setting sunglass dimensions
    print("Setting sunglass dimensions")
    eyeDis = abs(eyeCenters[0][0]-eyeCenters[1][0])*2.5
    sunglassScale = ((2.2*eyeDis)/sunglass.shape[1]) * 100
    sunglass = scaleImg(sunglass,sunglassScale)

    # Setting the sunglass start postion
    print("Setting sunglass position")
    yEye, xEye = int(2.5*eyeCenters[0][1]-0.25*eyeDis), int(2.5*eyeCenters[0][0]-0.5*eyeDis)
    cols,rows,channels = sunglass.shape
    roi = face[yEye:yEye+cols, xEye:xEye+rows]

    # Display
    plt.imshow(roi,"gray")
    plt.show()
    
    # Now create a mask of logo and create its inverse mask also
    print("Masking sunglass onto original image")
    img2gray = cv2.cvtColor(sunglass,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Display
    plt.imshow(mask_inv,"gray")
    plt.show()

    # Now black-out the area of the sunglass
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask)

    # Display
    plt.imshow(img1_bg,"gray")
    plt.show()

    # Take only region of sunglass from the sunglass image
    img2_fg = cv2.bitwise_and(sunglass,sunglass,mask = mask_inv)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    face[yEye:yEye+cols, xEye:xEye+rows] = dst

    cv2.imwrite('finalOutput.jpg', face)

    print("Done")
    face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
    plt.imshow(face)
    plt.show()
    
    return face 

def main():
    args = []
    for arg in sys.argv[1:]:
        args.append(arg)

    if len(args) == 0:
        args = ["imgL.jpg","imgR.jpg","sunglass.png",False]

    if args[3] == "True":  args[3] = True
    else: args[3] = False
    
    sunglassFilter(args[0], args[1], args[2], args[3])


if __name__ == "__main__":
    
    #!/usr/bin/python
    import sys
    import os
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt

    main()

    


