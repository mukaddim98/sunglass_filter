# Sunglass Filter
This is an OpenCV project that takes two disparity images of a face and places a sunglass on the face.

![final_output](https://github.com/mukaddim98/sunglass_filter/blob/main/finalOutput.jpg "Final Output")

The main program is in the ```sunglass_filter.py``` which has a function called ```sunglassFilter``` that takes in 4 parameters: image left path, image right path, sunglass image path and a parameter for background which can be either ```True``` or ```False```.

## How to run the code
To run the code open command line terminal and typr the following code given below.

__For image without background use:__ 

```py sunglass_filter.py imgL.jpg imgR.jpg sunglass.png False```

__For image with background use:__

```py sunglass_filter.py imgLwBG.jpg imgRwBG.jpg sunglass.png True```

The only difference in the code, is that when we use an image with a complicated background, we increase the threshold to make more pixels white to create the whitebackground, and use a higher K value for segmentation, because the image lighting might be poor.

## Important to note about input images

The main challenge of the project is taking the photo. As mentioned in the proposal, photos must be taken in a well-lit environment, ~60cm far from the camera, and each image taken ~5cm apart horizontally. The person must look straight into the camera and must not wear anything in the face. 

Addition to that, the background must be as still as possible. Better use a white backdrop like in the presentation. If background is used, then it is important that the background is very far, so that it's disparity is low. 


