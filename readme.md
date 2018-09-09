
# Toybox

These are various scripts I've made that aren't connected to anything else.
They don't have any other home, so they are here.

# Scripts

* ./vision/crop_image_on_faces.py

    This script uses OpenCV to detect faces in a photo and then crops the photo to cut out the space around the space shared by all faces.
    The user can specify the width and height of the output--it is useful for making more (semi)-intelligently making more meaningful
    thumbnails of photos.  Uses OpenCV's default face detection training set.  Accuracy is good but not perfect.
