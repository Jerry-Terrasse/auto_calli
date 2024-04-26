import cv2
import numpy as np
from shapely.geometry import LineString, box
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN

Line = tuple[int, int, int, int]

# 定义全局参数
THRESHOLD_FACTOR = 0.5
GAUSSIAN_BLUR_SIZE = (3, 3)
CANNY_THRESHOLD_LOW = 50
CANNY_THRESHOLD_HIGH = 100
HOUGH_THRESHOLD = 100
MIN_LINE_LENGTH_FACTOR = 0.5
MAX_LINE_GAP = 10
NORM_SIZE = (512, 512)

def display_image(img, delay: int = 1, title: str = 'Image'):
    cv2.imshow(title, img)
    cv2.waitKey(delay)

def load_image(file_path):
    return cv2.imread(file_path)

def compute_line_properties(line):
    angle = np.arctan2(line[3] - line[1], line[2] - line[0])
    center = ((line[0] + line[2]) // 2, (line[1] + line[3]) // 2)
    return angle, center

# def extend_line(line, width, height):
#     x1, y1, x2, y2 = line.bounds
#     if x1 == x2:
#         return LineString([(x1, 0), (x1, height)])
#     if y1 == y2:
#         return LineString([(0, y1), (width, y1)])
#     k = (y2 - y1) / (x2 - x1)
#     b = y1 - k * x1
#     x_a, y_a = 0, b
#     x_b, y_b = width, k * width + b
#     extended_line = LineString([(x_a, y_a), (x_b, y_b)])
#     return extended_line.intersection(box(0, 0, width, height))

def extend_line(line: Line, box_w: int, box_h: int) -> Line:
    x1, y1, x2, y2 = line
    if x1 == x2:
        return (x1, 0, x1, box_h)
    if y1 == y2:
        return (0, y1, box_w, y1)
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    x_a, y_a = 0, b
    x_b, y_b = box_w, k * box_w + b
    line_obj = LineString([(x_a, y_a), (x_b, y_b)])
    line_obj = line_obj.intersection(box(0, 0, box_w, box_h))
    return (int(line_obj.xy[0][0]), int(line_obj.xy[1][0]), int(line_obj.xy[0][1]), int(line_obj.xy[1][1]))
    

def process_image_for_lines(img, char_mask):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    display_image(gray_img, 0, title='Gray Image')

    # remove characters
    char_mask = cv2.dilate(char_mask, np.ones((3, 3), np.uint8), iterations=2)
    # gray_img[char_mask != 0] = 255
    
    blurred_img = cv2.GaussianBlur(gray_img, GAUSSIAN_BLUR_SIZE, 0)
    # display_image(blurred_img, 0, title='Blurred Image')
    edges = cv2.Canny(blurred_img, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)
    # display_image(edges, 0, title='Edges Image')
    edges[char_mask != 0] = 0
    # display_image(edges, 0, title='Edges Image without Characters')

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
    # display_image(edges, 0, title='Dilated and Eroded Edges Image')
    edges = cv2.erode(edges, kernel, iterations=3)
    # display_image(edges, 0, title='Dilated and Eroded Edges Image')
    return edges

def line_angle(line: Line) -> float: # range [0, pi)
    angle = np.arctan2(line[3] - line[1], line[2] - line[0])
    return angle if angle >= 0 else angle + np.pi

def line_v_or_h(line: Line, thresh: float = np.pi / 30) -> bool:
    angle = line_angle(line)
    return angle < thresh or angle > np.pi - thresh or np.pi / 2 - thresh < angle < np.pi / 2 + thresh

def find_lines(img, width, height, min_line_length=0):
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, HOUGH_THRESHOLD, minLineLength=min_line_length, maxLineGap=MAX_LINE_GAP)
    lines = [extend_line(line[0], width, height) for line in lines]
    lines = [line for line in lines if line_v_or_h(line)]
    return lines

def unique_lines(lines: list[Line], neighbor_size: float) -> list[Line]:
    dbscan = DBSCAN(eps=neighbor_size, min_samples=1)
    X = np.array(lines)
    clusters = dbscan.fit_predict(X)
    print(clusters)
    assert -1 not in clusters, 'Cluster -1 is reserved for noise'
    clustered_lines = [[] for _ in set(clusters)]
    for label, line in zip(clusters, lines):
        clustered_lines[label].append(line)
    unique_lines = []
    for cluster in clustered_lines:
        if len(cluster) == 1:
            unique_lines.append(cluster[0])
        else:
            cluster = np.array(cluster)
            mean_line = np.mean(cluster, axis=0)
            unique_lines.append(tuple(mean_line))
    return unique_lines

def get_char_mask(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # display_image(gray_img, 0, title='Gray Image')
    blurred_img = cv2.GaussianBlur(gray_img, GAUSSIAN_BLUR_SIZE, 0)
    # display_image(blurred_img, 0, title='Blurred Image')
    v_mean = np.mean(blurred_img)
    mask = cv2.threshold(blurred_img, v_mean * THRESHOLD_FACTOR, 255, cv2.THRESH_BINARY_INV)[1]
    # display_image(mask, 0, title='Mask Image')
    return mask

def determine_grid(lines: list[Line]) -> tuple[list[Line], list[Line], tuple[int, int]]:
    pass

def main():
    std_image_path = 'std_mu.png'
    exp_image_path = 'exp1.jpg'

    # Load and process standard image
    std_image = load_image(std_image_path)
    std_image_copy = std_image.copy()
    std_height, std_width = std_image.shape[:2]

    mask_std = get_char_mask(std_image)
    edges_std = process_image_for_lines(std_image, mask_std)
    lines_std = find_lines(edges_std, std_width, std_height, int(std_height * MIN_LINE_LENGTH_FACTOR))
    lines_std = unique_lines(lines_std, std_height / 5)

    # Load and process experimental image
    exp_image = load_image(exp_image_path)
    exp_image_copy = exp_image.copy()
    exp_height, exp_width = exp_image.shape[:2]

    mask_exp = get_char_mask(exp_image)
    edges_exp = process_image_for_lines(exp_image, mask_exp)
    lines_exp = find_lines(edges_exp, exp_width, exp_height, int(exp_height / 5 * MIN_LINE_LENGTH_FACTOR))
    lines_exp = unique_lines(lines_exp, exp_height / 5 / 5)
    lines_h, lines_v, (h, w) = determine_grid(lines_exp)

    # Draw lines on the images for visualization
    for line in lines_std:
        x1, y1, x2, y2 = map(int, line)
        cv2.line(std_image_copy, (x1, y1), (x2, y2), (0, 255, 0), 1)
    for line in lines_exp:
        x1, y1, x2, y2 = map(int, line)
        cv2.line(exp_image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the images
    cv2.imshow('Standard Image with Lines', std_image_copy)
    cv2.imshow('Experimental Image with Lines', exp_image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
