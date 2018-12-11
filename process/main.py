import process.model_train as mt
import cv2


def similarity(img1=None, img2=None):
    assert img1 is not None and img2 is not None
    dim = 148
    img1 = cv2.resize(img1, (dim, dim))
    img2 = cv2.resize(img2, (dim, dim))
