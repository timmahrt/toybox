'''
Created on Sep 8, 2018

Use autocropFaces() to crop out the material around faces in an image,
where the faces are automatically detected.

See the bottom for an example use script.

Used this as a starting reference point:
https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html

@author: tmahrt
'''

import os
from os.path import join

import cv2
from matplotlib import pyplot as plt
from PIL import Image

TRAINING_DATA_PATH = '/opt/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'


class NoFacesException(Exception):
    
    def __init__(self, fn):
        super(NoFacesException, self).__init__()
        self.fn = fn
    
    def __str__(self):
        errStr = ("ERROR: Could not find faces in file `%s` with "
                  "training data: \n`%s`\n  Please try again with a different "
                  "file, or different training set.")
        return errStr % (self.fn, TRAINING_DATA_PATH)
        
    
class FaceRecognizer():
    
    def __init__(self):
        self.recognizer = cv2.CascadeClassifier(TRAINING_DATA_PATH)

    def recognize(self, imgFn):
        gray = cv2.imread(imgFn, 0)
        
        faces = self.recognizer.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            raise NoFacesException(imgFn)
        
        return faces

     
def outputDebug(imgFn,
                faces,
                faceRegion=None,
                helperRegion=None,
                finalCropRegion=None):
    
    img = cv2.imread(imgFn)
    
    # The list of faces
    for face in faces:
        _drawRectangle(img, face, (255, 0, 0))
    
    # All the faces fit tightly in this space
    if faceRegion is not None:
        _drawRectangle(img, faceRegion, (0, 0, 255))
    
    # I used this to see various intermediate stages
    if helperRegion is not None:
        _drawRectangle(img, helperRegion, (0, 255, 0))
    
    # The final cropping region
    if finalCropRegion is not None:
        _drawRectangle(img, finalCropRegion, (255, 255, 0))
    
    img = _convertBgrToRGB(img)
    plt.imshow(img)
    plt.show()


def _convertBgrToRGB(img):
    # https://stackoverflow.com/questions/15072736/extracting-a-region-from-an-image-using-slicing-in-python-opencv/15074748#15074748
    return img[:, :, ::-1]


def _drawRectangle(img, xywh, color):
    x, y, w, h = xywh
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)


def encapsulateSubsquares(regionList):
    '''
    Given a list of squares, return a square that tightly fits all subsquares
    
    Input is a list of the form [(x, y, w, h), () ]
    Output is the (x, y, w, h) that wholly includes all input
    '''
    newRegionList = [(x, y, x + w, y + h) for x, y, w, h in regionList]
    x0List, y0List, x1List, y1List = zip(*newRegionList)
    
    x0 = min(x0List)
    y0 = min(y0List)
    x1 = max(x1List)
    y1 = max(y1List)
    
    return [x0, y0, x1 - x0, y1 - y0]


def modifyAspectRatio(sourceXYWH, targetRatio):
    '''
    Changes the ratio of the input square to be that of the target ratio
    '''
    sourceRatio = sourceXYWH[2] / sourceXYWH[3]
    if targetRatio > sourceRatio:
        newX1 = int(sourceXYWH[3] * targetRatio)
        returnXYWH = [sourceXYWH[0], sourceXYWH[1],
                      newX1, sourceXYWH[3]]
    else:
        newY1 = int(sourceXYWH[2] / targetRatio)
        returnXYWH = [sourceXYWH[0], sourceXYWH[1],
                      sourceXYWH[2], newY1]

    return returnXYWH


def relativeRecenter(sourceXYWH, targetXYWH):
    '''
    Centers a square with respect to the center of a different square
    '''
    targetXCenter = targetXYWH[0] + (targetXYWH[2] / 2.0)
    targetYCenter = targetXYWH[1] + (targetXYWH[3] / 2.0)
    
    newX = int(targetXCenter - (sourceXYWH[2] / 2.0))
    newY = int(targetYCenter - (sourceXYWH[3] / 2.0))

    return (newX, newY, sourceXYWH[2], sourceXYWH[3])


def keepInsideImage(sourceXYWH, imageWH):
    '''
    Forces a square to be within the image that contains it
    '''
    
    left = sourceXYWH[0]
    right = sourceXYWH[0] + sourceXYWH[2]
    
    top = sourceXYWH[1]
    bottom = sourceXYWH[1] + sourceXYWH[3]
    
    newLeft = left
    if left < 0 and right > imageWH[0]:
        newLeft = (imageWH[0] - right)
    elif left < 0:
        newLeft = 0
    elif right > imageWH[0]:
        newLeft = imageWH[0] - sourceXYWH[2]
    
    newTop = top
    if top < 0 and bottom > imageWH[1]:
        newTop = imageWH[1] / 2.0 - sourceXYWH[3]
    elif top < 0:
        newTop = 0
    elif bottom > imageWH[1]:
        newTop = imageWH[1] - sourceXYWH[3]
    
    return [int(newLeft), int(newTop), sourceXYWH[2], sourceXYWH[3]]


def enforceMinSize(sourceXYWH, targetWH, imgWH):
    '''
    Increase the crop region to the target, but don't exceed the img dimensions
    '''
    newW = max((targetWH[0], sourceXYWH[2]))
    newH = max((targetWH[1], sourceXYWH[3]))
    
    newW = min((imgWH[0], newW))
    newH = min((imgWH[1], newH))

    return (sourceXYWH[0], sourceXYWH[1], newW, newH)


def autocropFaces(fn, outputFN, recognizer, targetWH=None, debug=False):
    '''
    Will crop an image based on all of the faces it automatically detects
    
    targetWH: e.g. (300, 200); if specified, it the output will that size.
              The area around the detected heads will be enlarged to permit
              the necessary aspect ratio before scaling occurs.  If the image
              is smaller than the target, whitespace will be filled in.
    debug: if True, an image will pop up showing detected faces and the
              region that will be cropped.  The image must be closed before
              the code will continue
    '''
    faceList = recognizer.recognize(fn)
    faceRegion = encapsulateSubsquares(faceList)
    
    img = Image.open(fn)
    imgWH = (img.width, img.height)
    if targetWH is not None:
        sizedFaceRegion = enforceMinSize(faceRegion, targetWH, imgWH)
        proportionedFaceRegion = modifyAspectRatio(sizedFaceRegion,
                                                   targetWH[0] / targetWH[1])
        
        regionToCenterIn = relativeRecenter(sizedFaceRegion,
                                            faceRegion)
        adjustedFaceRegion = relativeRecenter(proportionedFaceRegion,
                                              regionToCenterIn)
        adjustedFaceRegion = keepInsideImage(adjustedFaceRegion, imgWH)

        # If the crop region is smaller than the targetWH, fill in
        # the empty space with a white background
        newImg = Image.new('RGB',
                           (adjustedFaceRegion[2], adjustedFaceRegion[3]),
                           (255, 255, 255))
        newImg.paste(img, (-adjustedFaceRegion[0], -adjustedFaceRegion[1]))
        img = newImg
        
        if debug is True:
            outputDebug(fn, faceList, faceRegion, sizedFaceRegion,
                        finalCropRegion=adjustedFaceRegion)
    else:
        img = img.crop(faceRegion)
    
    if targetWH is not None:
        img = img.resize(targetWH)
    
    img.save(outputFN)


# Example use
if __name__ == "__main__":
    
    def getThumbnailName(fn):
        name, ext = os.path.splitext(fn)
        return name + "_thumbnail" + ext
    
    inputPath = os.path.abspath("../data/faces/")
    outputPath = os.path.abspath("../data/faces/output")
    targetWH = (300, 200)
    
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    
    _recognizer = FaceRecognizer()
    for _fn in os.listdir(inputPath):
        if ".jpg" not in _fn:
            continue
        inputFn = join(inputPath, _fn)
        outputFn = join(outputPath, getThumbnailName(_fn))
        try:
            autocropFaces(inputFn, outputFn, _recognizer, targetWH, debug=True)
        except NoFacesException:
            print("No faces in: " + inputFn)
            continue
