import numpy as np
import cv2 as cv
import requests


def url2img(urls, RGB = False):
    resp = requests.get(urls, stream=True).raw
    img = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv.imdecode(img, cv.IMREAD_COLOR)
    if RGB :
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def split(word):
    return [char for char in word]


def resize_image(img_src, max_size):
    height, width = img_src.shape[:2] # Get the original image dimensions
    # Calculate the new dimensions while maintaining the aspect ratio
    if width > height:
        new_width = max_size
        new_height = int(max_size * (height / width))
    else:
        new_height = max_size
        new_width = int(max_size * (width / height))

    return cv.resize(img_src, (new_width, new_height), cv.INTER_LINEAR)


def rescale_image(img_src, scale):
    height, width = img_src.shape[:2] # Get the original image dimensions
    # Calculate the new dimensions based on the scale factor
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv.resize(img_src, (new_width, new_height), cv.INTER_LINEAR)