#!/usr/bin/env python

import sys
import cv2 as cv
import numpy as np
import math


# Program: similarity.py
# Objective: To compare simiarity between two images of same shape
# Author: jumail@utm.my
# Date  : 16 Jan 2023
# 
# This program is to be run from command line
# 
# Syntax: 
# 
# On Command Prompt terminal (CMD):
#  similarity.py image1 image2 threshold
# 
# On Powershell or Bash Terminal:
# ./similarity.py image1 image2 threshold
# 
# Important notes: 
#  To allow your computer to execute a python script from the command line, 
#   you must "Associate" the  .py file extension to Python interpreter


# Default values
DEFAULT_THRESHOLD = 10
BORDER_SIZE = 15
CAPTION_SIZE = 50


# Function to calculate the difference between two images 

def similarity(srcImage1, srcImage2, threshold):
    
    # Convert the images into float32 to prevent truncation 
    #  if subtraction results in negative value
    image1 = srcImage1.copy().astype(np.float32)  
    image2 = srcImage2.copy().astype(np.float32)
    
    # Create the base matrix for the calculation of pixels that differs 
    #   larger than the threshold

    height, width, channel = image1.shape
    result = np.zeros((height, width,channel), np.uint8)
    
    # Assign 1 to cells whose differences between the images 
    #   larger than the threshold, to prepare for the sum up later
   
    result[abs(image1 - image2 ) <= threshold] = 1
    simRate = result.sum() / (height*width*channel)

    return math.floor(10000*simRate + 0.5)/100

def processCommandLine():
    argc = len(sys.argv) - 1

    if argc < 2:
        print('\n** Error: need two image files as input\n')
        print('Syntax (Bash and PowerShell): ./similarity.py image1 image2 [threshold]')
        print(
            'Syntax (Command Prompt): similarity.py image1 image2 [threshold]')

        print('\n  The two images are the Noised Image and the Reference Image. Not necessarily in order.')
        print(
            f'  Threshold is optional. If not specified, the default is {DEFAULT_THRESHOLD}.')
        print('\nExample:  ./similarity.py ref_image.jpg  restored.jpg')
        sys.exit(1)
        return

    if argc == 2:
        _, file1, file2 = sys.argv
        return file1, file2, DEFAULT_THRESHOLD

    _, file1, file2, threshold = sys.argv
    threshold = int(threshold)
    return file1, file2, threshold


def display(image1, image2, sim, thresh):
    height, width, channel = image1.shape

    border = np.ones((height, BORDER_SIZE, channel), np.uint8)*255
    caption = np.ones((CAPTION_SIZE, 2*width+BORDER_SIZE,channel), np.uint8)*255
    caption = cv.putText(caption, f'Similarity: {sim}% (Threshold={thresh})', (10, 20), cv.FONT_HERSHEY_SIMPLEX,
                         0.7, (0, 0, 0), 2, cv.LINE_AA)

    image = np.concatenate((image1, border, image2), axis=1)
    image = np.concatenate((image, caption), axis=0,)

    cv.imshow('Image Similarity', image)


def main():
    images = [None, None]
    heights = [None, None]
    widths = [None, None]
    channels = [None, None]
    files = [None, None]

    files[0], files[1], threshold = processCommandLine()

    for i in range(len(images)):
        images[i] = cv.imread(files[i])
        # images[i] = cv.cvtColor(images[i], cv.COLOR_BGR2GRAY)
        heights[i], widths[i], channels[i]  = images[i].shape


    if (heights[0] != heights[1]) or (widths[0] != widths[1]) or (channels[0] != channels[1]):
        print('\n** Error: Both images must be of same shape\n')
        sys.exit(2)

    sim = similarity(images[0], images[1], threshold)


    display(images[0], images[1], sim, threshold)

    print(f'Similarity: {sim}%')

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
