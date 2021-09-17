import torch, os, shutil
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import colorsys, random
import sys, imutils
from pathlib import Path


class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


class Logger:
    def __init__(self, filename):
        self.console  = sys.stdout
        self.filename = filename

    def write(self, message):
        self.console.write(message)
        with open(self.filename, "a+") as f:
            f.write(message)

    def flush(self):
        self.console.flush()
        # self.file.flush()


def draw_box(image, box: list, box_color: tuple = (255,0,0)):
    """
    :param image: Image array with shape[0] as height, shape[1] as width
    :param box: xmin, ymin, xmax, ymax
    :param box_color: color of the box
    :return: image with drawn box
    """
    image_h, image_w = image.shape[0], image.shape[1]
    bbox_thick = int(0.6 * (image_h + image_w) / 1000)
    if bbox_thick < 1: bbox_thick = 1
    box = [ int(x) for x in box ]
    x1, y1, x2, y2 = box

    # put object rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, bbox_thick*2)

    return image


def draw_text(image, cell_data: list, result_path: str,
              column_headers: list, row_headers: list = None, row_colors: list = None
              ):
    """
    :param image:
    :param cell_data:
    :param result_path:
    :param column_headers:
    :param row_headers:
    :param row_colors:
    :return:
    """

    # plot image
    plt.imshow(image)

    # styling columns and rows
    the_table = plt.table(cellText=cell_data,
                          rowLabels= row_headers,
                          colLabels=column_headers,
                          rowColours= row_colors,
                          rowLoc='right',
                          loc='bottom')

    # Scaling table
    the_table.scale(1, 1.5)

    # Hide axis
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Hide axis border
    plt.box(on=None)
    title_text = "Results"

    # set title
    plt.suptitle(title_text)

    # Create image. plt.savefig ignores figure edge and face colors, so map them.
    #todo
    fig = plt.gcf()
    plt.savefig( result_path,
                # bbox='tight',
                edgecolor=fig.get_edgecolor(),
                facecolor=fig.get_facecolor(),
                dpi=150
                )



def visualize_detections(image, scores, boxes, labels, result_path):

    # todo: need to take as input
    total_labels = 100

    # random colors generation
    hsv_tuples = [(1.0 * x / 100, 1., 1.) for x in range(total_labels)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    # initialize
    scores_per_img = []
    labels_per_image = []

    for score,box,label in zip(scores,boxes,labels):

        # label color
        box_color = [x*255 for x in colors[label] ]

        # draw box
        draw_box(image, box, box_color)

        # # read mask
        # mask = np.round(mask)
        #
        # # draw mask on image
        # image[np.where(mask != 0)] = label

        labels_per_image.append(label)
        scores_per_img.append(score)


    # Draw text: Add a table at the bottom of the axes
    column_headers = ["labels", "scores"]
    cell_data = [ [str(label), str(score)] for label,score in zip(labels_per_image,scores_per_img) ]
    row_colors = [ colors[label] for label in labels_per_image ]

    # draw_text(image, cell_data = cell_data,result_path=result_path,
    #                  column_headers = column_headers,
    #                  row_colors= row_colors)

    plt.imsave(result_path, image)


def limit_to_4_corners(points, x_min, y_min, x_max, y_max):
    """
    From the given points find 4 points that best fit the corners
    :param points:
    :param x_min:
    :param y_min:
    :param x_max:
    :param y_max:
    :return:
    """
    # find 4 corner points of mask image
    top_left_corner_points = []
    top_right_corner_points = []
    bottom_left_corner_points = []
    bottom_right_corner_points = []

    for point in points:
        if point[1] <= (x_min + int((x_max - x_min) / 4)) and point[0] <= (y_min + int((y_max - y_min) / 2)):
            top_left_corner_points.append(point)
        if point[1] >= (x_max - int((x_max - x_min) / 4)) and point[0] <= (y_min + int((y_max - y_min) / 2)):
            top_right_corner_points.append(point)
        if point[1] >= (x_max - int((x_max - x_min) / 4)) and point[0] >= (y_max - int((y_max - y_min) / 2)):
            bottom_right_corner_points.append(point)
        if point[1] <= (x_min + int((x_max - x_min) / 4)) and point[0] >= (y_max - int((y_max - y_min) / 2)):
            bottom_left_corner_points.append(point)
    if top_left_corner_points == []:
        for point in points:
            if point[1] <= (x_min + int((x_max - x_min) / 2)) and point[0] <= (y_min + int(y_max - y_min)):
                top_left_corner_points.append(point)
    if top_right_corner_points == []:
        for point in points:
            if point[1] >= (x_max - int((x_max - x_min) / 2)) and point[0] <= (y_min + int(y_max - y_min)):
                top_right_corner_points.append(point)
    if bottom_right_corner_points == []:
        for point in points:
            if point[1] >= (x_max - int((x_max - x_min) / 2)) and point[0] >= (y_max - int(y_max - y_min)):
                bottom_right_corner_points.append(point)
    if bottom_left_corner_points == []:
        for point in points:
            if point[1] <= (x_min + int((x_max - x_min) / 2)) and point[0] >= (y_max - int(y_max - y_min)):
                bottom_left_corner_points.append(point)

    top_left_corner_points = np.array(top_left_corner_points)
    top_right_corner_points = np.array(top_right_corner_points)
    bottom_right_corner_points = np.array(bottom_right_corner_points)
    bottom_left_corner_points = np.array(bottom_left_corner_points)

    # print("top_left_corner_points",top_left_corner_points.shape)
    top_left_point = [np.min(top_left_corner_points[:, 1]), np.min(top_left_corner_points[:, 0])]
    bottom_right_point = [np.max(bottom_right_corner_points[:, 1]), np.max(bottom_right_corner_points[:, 0])]
    top_right_point = [np.max(top_right_corner_points[:, 1]), np.min(top_right_corner_points[:, 0])]
    bottom_left_point = [np.min(bottom_left_corner_points[:, 1]), np.max(bottom_left_corner_points[:, 0])]

    return top_left_point, bottom_right_point, top_right_point, bottom_left_point


def detect_corners(mask_img):
    """
    detect 4 corners of the given mask using opencv strategies
    :param mask_img:
    :return:
    """
    mask_img = np.array(mask_img, dtype=np.uint8)

    # edge detection
    ret2, thresold = cv2.threshold(mask_img.copy(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # contour
    contours, hierarchy = cv2.findContours(thresold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours((contours, 0))
    max_area_contour = max(contours, key=cv2.contourArea)
    # print("total contour points", len(max_area_contour))

    # approxPolyDP : finds minimum points to fit contour
    perimeter = cv2.arcLength(max_area_contour, True)
    approximatedPoints = cv2.approxPolyDP(max_area_contour, 0.02 * perimeter, True).reshape(-1,2)
    # print("approximated contour points", len(approximatedPoints))

    # if final points are exactly 4, return the 4 points
    if len(approximatedPoints) == 4:
        return approximatedPoints

    # if approx points is not 4, use harris corner strategy
    elif len(approximatedPoints) < 4 or len(approximatedPoints) > 4:
        # Harris corners
        approximatedPoints = cv2.goodFeaturesToTrack(thresold.copy(), 20, 0.01, 10).reshape(-1,2)


    # reorder points x,y to y,x
    approximatedPoints = approximatedPoints[:, ::-1]

    # limit to 4 points only
    y, x = np.where(thresold == 255)
    x_min, x_max, y_min, y_max = np.min(x), np.max(x), np.min(y), np.max(y)
    approximatedPoints = limit_to_4_corners(approximatedPoints, x_min, y_min, x_max, y_max)

    # # visualize
    # temp_img = np.zeros( mask_img.shape, dtype=np.uint8 )
    # for (x,y) in approximatedPoints:
    #     cv2.circle(temp_img, (x, y), 8, (255, 0, 0), 3)
    # plt.imshow(temp_img)
    # plt.title( len(approximatedPoints) )
    # plt.show()

    return approximatedPoints


def extract_warped_masks(image, masks):
    """
    Extract the mask and return the four point translated mask (numpy)
    :param image:
    :param masks:
    :return:
    """
    warped_masks = []
    for mask in masks:

        img = mask[0].cpu().detach().numpy()
        img = np.round(img)

        # extract 4 corner points
        top_left_point, top_right_point, bottom_right_point, bottom_left_point = detect_corners(img)
        pts = np.array([top_left_point, top_right_point, bottom_right_point, bottom_left_point])

        # transform mask to rectangle
        # for (x, y) in pts:
        #     cv2.circle(image, (x, y), 10, (255, 0, 0), 5)
        # save_dir = "/home/sohoa1/rammy/datasets/digital_meter/yolact/mask_annotations_yolact/visualize_results/corner_extraction"
        # fnames = [int(Path(x).stem) for x in os.listdir(save_dir)]
        # if len(fnames) == 0:
        #     filename = "1.jpg"
        # else:
        #     filename = str(max(fnames) + 1) + ".jpg"
        # plt.imsave(os.path.join(save_dir, filename), image)
        # plt.imshow(image)
        # plt.show()

        warped = four_point_transform(image, pts)

        # rotate
        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        # plt.imshow(warped)
        # plt.show()
        warped_masks.append(warped)

    return warped_masks


def four_point_transform(image, pts):
    """

    :param image: numpy array
    :param pts: an array of format => array([top_left_point, top_right_point, bottom_right_point, bottom_left_point]) , shape: 4x2
    :return:
    """
    if isinstance(image, torch.Tensor): image = image.cpu().detach().numpy()
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective( image , M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

