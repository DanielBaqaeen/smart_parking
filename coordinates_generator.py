import cv2 as open_cv
import numpy as np

COLOR_RED = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)


def draw_contours(image,
                  coordinates,
                  label,
                  font_color,
                  border_color=COLOR_RED,
                  line_thickness=1,
                  font=open_cv.FONT_HERSHEY_SIMPLEX,
                  font_scale=0.5):
    open_cv.drawContours(image,
                         [coordinates],
                         contourIdx=-1,
                         color=border_color,
                         thickness=2,
                         lineType=open_cv.LINE_8)
    moments = open_cv.moments(coordinates)

    center = (int(moments["m10"] / moments["m00"]) - 3,
              int(moments["m01"] / moments["m00"]) + 3)

    open_cv.putText(image,
                    label,
                    center,
                    font,
                    font_scale,
                    font_color,
                    line_thickness,
                    open_cv.LINE_AA)


class CoordinatesGenerator:
    KEY_RESET = ord("r")
    KEY_QUIT = ord("q")

    def __init__(self, image, output, color):
        self.output = output
        self.caption = image
        self.color = color

        self.image = open_cv.imread(image).copy()
        self.click_count = 0
        self.ids = 1
        self.coordinates = []

        open_cv.namedWindow(self.caption, open_cv.WINDOW_GUI_EXPANDED)
        open_cv.setMouseCallback(self.caption, self.__mouse_callback)

    def generate(self):
        while True:
            open_cv.imshow(self.caption, self.image)
            key = open_cv.waitKey(0)

            if key == CoordinatesGenerator.KEY_RESET:
                self.image = self.image.copy()
            elif key == CoordinatesGenerator.KEY_QUIT:
                break
        open_cv.destroyWindow(self.caption)

    def __mouse_callback(self, event, x, y, flags, params):

        if event == open_cv.EVENT_LBUTTONDOWN:
            self.coordinates.append((x, y))
            self.click_count += 1

            if self.click_count >= 4:
                self.__handle_done()

            elif self.click_count > 1:
                self.__handle_click_progress()

        open_cv.imshow(self.caption, self.image)

    def __handle_click_progress(self):
        open_cv.line(self.image, self.coordinates[-2], self.coordinates[-1], (255, 0, 0), 1)

    def __handle_done(self):
        open_cv.line(self.image,
                     self.coordinates[2],
                     self.coordinates[3],
                     self.color,
                     1)
        open_cv.line(self.image,
                     self.coordinates[3],
                     self.coordinates[0],
                     self.color,
                     1)

        self.click_count = 0

        coordinates = np.array(self.coordinates)

        height, width, c = self.image.shape

        # normalize coordinates:
        # compare y1 and y2, smallest ymin
        if self.coordinates[0][1] < self.coordinates[1][1]:
            Ymin = coordinates[0][1] / height
        else:
            Ymin = self.coordinates[1][1] / height

        # compare x1 and x4, smallest is xmin
        if self.coordinates[0][0] < self.coordinates[3][0]:
            Xmin = self.coordinates[0][0] / width
        else:
            Xmin = self.coordinates[3][0] / width

        # compare y3 and y4, largest ymax
        if self.coordinates[2][1] > self.coordinates[3][1]:
            Ymax = self.coordinates[2][1] / height
        else:
            Ymax = self.coordinates[3][1] / height

        # compare x2 and x3, largest is xmax
        if self.coordinates[1][0] > self.coordinates[2][0]:
            Xmax = self.coordinates[1][0] / width
        else:
            Xmax = self.coordinates[2][0] / width

        # self.output.write("-\n          id: " + str(self.ids) + "\n          coordinates: [" +
        #                   "[" + str(self.coordinates[0][0]) + "," + str(self.coordinates[0][1]) + "]," +  # x1,y1
        #                   "[" + str(self.coordinates[1][0]) + "," + str(self.coordinates[1][1]) + "]," +  # x2,y2
        #                   "[" + str(self.coordinates[2][0]) + "," + str(self.coordinates[2][1]) + "]," +  # x3,y3
        #                   "[" + str(self.coordinates[3][0]) + "," + str(self.coordinates[3][1]) + "]]\n")  # x4,y4

        self.output.write(
            "- id: " + str(self.ids) + "\n  Ymin: " + str(Ymin) + "\n  Xmin: " + str(Xmin) + "\n  Ymax: " + str(
                Ymax) + "\n  Xmax: " + str(Xmax) + "\n\n")

        draw_contours(self.image, coordinates, str(self.ids), COLOR_WHITE)

        for i in range(0, 4):
            self.coordinates.pop()

        self.ids += 1
