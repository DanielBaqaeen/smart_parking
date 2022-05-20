import argparse
import logging
from coordinates_generator import CoordinatesGenerator
import cv2
import time

COLOR_RED = (255, 0, 0)



def lab():
    camera_port = 0
    camera = cv2.VideoCapture(camera_port)
    time.sleep(0.1)  # If you don't wait, the image will be dark
    return_value, image = camera.read()
    cv2.imwrite("parking_lot.jpg", image)
    del (camera)  # so that others can use the camera as soon as possible


def generate():
    logging.basicConfig(level=logging.INFO)

    image_file = "parking_lotjpg"
    data_file = "coordinates_1.yml"

    if image_file is not None:
        with open(data_file, "w+") as points:
            generator = CoordinatesGenerator(image_file, points, COLOR_RED)
            generator.generate()


lab()
generate()
