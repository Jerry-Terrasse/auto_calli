import cv2
import numpy as np
from shapely import geometry
from shapely.geometry import Point, LineString

# def load_std(fname: str) -> cv2.Mat: # Mask
#     pass

def line_angle(line: list[int]) -> float: # range(0, pi)
    ang = np.arctan2(line[3] - line[1], line[2] - line[0])
    return ang if ang >= 0 else ang + np.pi

def line_center(line: list[int]) -> tuple[int, int]:
    return (line[0] + line[2]) // 2, (line[1] + line[3]) // 2

def line_extend(line: LineString, W: int, H: int) -> LineString:
    # extend the line to the edge of the image
    x1, y1, x2, y2 = line.bounds
    if x1 == x2:
        return LineString([(x1, 0), (x1, H)])
    if y1 == y2:
        return LineString([(0, y1), (W, y1)])
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    # if 0 <= b <= H:
    #     return LineString([(0, b), (W, k * W + b)])
    # if 0 <= k * H + b <= W:
    #     return LineString([(b, 0), ((H - b) / k, H)])
    # return line

    x_a, y_a = 0, b
    x_b, y_b = W, k * W + b
    line_ = LineString([(x_a, y_a), (x_b, y_b)])
    line_ = line_.intersection(geometry.box(0, 0, W, H))
    return line_

def line_v_or_h(line: list[int]) -> bool:
    a = line_angle(line)
    return a < np.pi/20 or a > np.pi*19/20 or np.pi*9/20 < a < np.pi*11/20
    # return a < np.pi/10 or a > np.pi*9/10 or np.pi*4/10 < a < np.pi*6/10

def line_v(line: list[int]) -> bool:
    a = line_angle(line)
    return np.pi/4 < a < np.pi*3/4

def line_h(line: list[int]) -> bool:
    a = line_angle(line)
    return a < np.pi/4 or a > np.pi*3/4

def get_mask_single(img: cv2.Mat) -> cv2.Mat:
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_img = hsv_img[:, :, 2]
    mean_v = np.mean(v_img)
    bin_img = cv2.threshold(v_img, mean_v/2, 255, cv2.THRESH_BINARY_INV)[1]
    return bin_img

def load_std():
    img = cv2.imread('std_mu.png')
    img_original = img.copy()
    H, W = img.shape[:2]
    cv2.imshow('img', img); cv2.waitKey(1)
    print(img.shape)
    # mean_color = np.mean(img, axis=(0, 1))
    # print(mean_color)
    
    # mean_img = np.full_like(img, mean_color)
    # cv2.imshow('mean_img', mean_img); cv2.waitKey(0)
    
    # detect grid
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_img = hsv_img[:, :, 2]
    # cv2.imshow('v_img', v_img); cv2.waitKey(0)
    mean_v = np.mean(v_img)
    print(mean_v)
    bin_img = cv2.threshold(v_img, mean_v, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imshow('bin_img', bin_img); cv2.waitKey(1)

    # erode
    # kernel = np.ones((3, 3), np.uint8)
    # bin_img = cv2.erode(bin_img, kernel, iterations=1)
    # bin_img = cv2.dilate(bin_img, kernel, iterations=1)
    # cv2.imshow('erode', bin_img); cv2.waitKey(0)

    # hough line
    lines = cv2.HoughLinesP(bin_img, 1, np.pi/1800, 100, minLineLength=W//2, maxLineGap=10)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        print(x1)
        # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imshow('line', img); cv2.waitKey(0)
    
    # find the left-most and right-most vertical line
    vertical_lines = [line for line in lines if np.pi/4 < line_angle(line[0]) < np.pi*3/4]
    left_line = min(vertical_lines, key=lambda line: line_center(line[0])[0])
    right_line = max(vertical_lines, key=lambda line: line_center(line[0])[0])

    horizontal_lines = [line for line in lines if line_angle(line[0]) < np.pi/4 or line_angle(line[0]) > np.pi*3/4]
    top_line = min(horizontal_lines, key=lambda line: line_center(line[0])[1])
    bottom_line = max(horizontal_lines, key=lambda line: line_center(line[0])[1])
    
    # for line in [left_line, right_line, top_line, bottom_line]:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imshow('line', img); cv2.waitKey(0)

    # find the intersection of the four lines
    left_line = geometry.LineString([left_line[0][:2], left_line[0][2:]])
    right_line = geometry.LineString([right_line[0][:2], right_line[0][2:]])
    top_line = geometry.LineString([top_line[0][:2], top_line[0][2:]])
    bottom_line = geometry.LineString([bottom_line[0][:2], bottom_line[0][2:]])
    left_line, right_line, top_line, bottom_line = [line_extend(line, W, H) for line in [left_line, right_line, top_line, bottom_line]]
    for line in [left_line, right_line, top_line, bottom_line]:
        x1, y1, x2, y2 = line.bounds
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
    cv2.imshow('line', img); cv2.waitKey(1)
    
    left_top = left_line.intersection(top_line)
    left_bottom = left_line.intersection(bottom_line)
    right_top = right_line.intersection(top_line)
    right_bottom = right_line.intersection(bottom_line)
    print(left_top, left_bottom, right_top, right_bottom)
    
    for point in [left_top, left_bottom, right_top, right_bottom]:
        cv2.circle(img, (int(point.x), int(point.y)), 5, (0, 255, 0), -1)
    cv2.imshow('line', img); cv2.waitKey(1)

    # normalize box to a 512x512 square using Homography
    pts_src = np.array([pt.xy for pt in [left_top, left_bottom, right_bottom, right_top]], dtype=np.float32)
    pts_dst = np.array([[0, 0], [0, 512], [512, 512], [512, 0]], dtype=np.float32)
    h, status = cv2.findHomography(pts_src, pts_dst)
    print(h, status)
    # assert status == 0
    
    img_norm = cv2.warpPerspective(img_original, h, (512, 512))
    cv2.imshow('img_norm', img_norm); cv2.waitKey(1)

    # inverse
    # h_inv = np.linalg.inv(h)
    # img_inv = cv2.warpPerspective(img_norm, h_inv, (W, H))
    # cv2.imshow('img_inv', img_inv); cv2.waitKey(0)
    
    std_mask = get_mask_single(img_norm)
    return img_original, img_norm, h, std_mask
    
if __name__ == '__main__':
    std_ori, std_norm, std_h, std_mask = load_std()
    cv2.imshow('std_mask', std_mask); cv2.waitKey(1)
    
    # load exp image
    img = cv2.imread('exp3_crop2.jpg')
    img_original = img.copy()
    # cv2.imshow('exp', img); cv2.waitKey(1)
    
    # blur
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    H, W = img.shape[:2]

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_img = hsv_img[:, :, 2]
    # cv2.imshow('v_img', v_img); cv2.waitKey(0)
    
    # local equalhist
    # clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    # v_img = clahe.apply(v_img)
    # cv2.imshow('v_img', v_img); cv2.waitKey(0)
    
    mean_v = np.mean(v_img)
    bin_img = cv2.threshold(v_img, mean_v, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imshow('bin_img', bin_img); cv2.waitKey(1)
    
    # kernel = np.ones((3, 3), np.uint8)
    # bin_img = cv2.erode(bin_img, kernel, iterations=0)
    # bin_img = cv2.dilate(bin_img, kernel, iterations=1)
    # cv2.imshow('erode', bin_img); cv2.waitKey(0)

    edge = cv2.Canny(bin_img, 50, 100)
    cv2.imshow('edge', edge); cv2.waitKey(0)
    
    kernel = np.ones((3, 3), np.uint8)
    edge = cv2.dilate(edge, kernel, iterations=2)
    edge = cv2.erode(edge, kernel, iterations=2)
    cv2.imshow('edge', edge); cv2.waitKey(1)
    
    # hough line
    lines = cv2.HoughLinesP(edge, 1, np.pi/1800, 0, minLineLength=H//2.5, maxLineGap=10)
    lines = [line for line in lines if line_v_or_h(line[0])]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imshow('line', img); cv2.waitKey(1)
    
    grid_w = 100 # TODO
    v_lines = [line for line in lines if line_v(line[0])]
    v_lines.sort(key=lambda line: line_center(line[0])[0])
    # v_lines_uniq = [v_lines[i] for i in range(len(v_lines)) if i == 0 or line_center(v_lines[i][0])[0] - line_center(v_lines[i-1][0])[0] > grid_w]
    v_lines_uniq = []
    for i in range(len(v_lines)):
        if i == 0 or line_center(v_lines[i][0])[0] - line_center(v_lines_uniq[-1][0])[0] > grid_w:
            v_lines_uniq.append(v_lines[i])
    # for line in v_lines_uniq:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # cv2.imshow('line', img); cv2.waitKey(1)
    v_lines_uniq = [line_extend(geometry.LineString([line[0][:2], line[0][2:]]), W, H) for line in v_lines_uniq]
    
    grid_h = 120 # TODO
    h_lines = [line for line in lines if line_h(line[0])]
    h_lines.sort(key=lambda line: line_center(line[0])[1])
    # h_lines_uniq = [h_lines[i] for i in range(len(h_lines)) if i == 0 or line_center(h_lines[i][0])[1] - line_center(h_lines[i-1][0])[1] > grid_h]
    h_lines_uniq = []
    for i in range(len(h_lines)):
        if i == 0 or line_center(h_lines[i][0])[1] - line_center(h_lines_uniq[-1][0])[1] > grid_h:
            h_lines_uniq.append(h_lines[i])
    # for line in h_lines_uniq:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # cv2.imshow('line', img); cv2.waitKey(1)
    h_lines_uniq = [line_extend(geometry.LineString([line[0][:2], line[0][2:]]), W, H) for line in h_lines_uniq]
    print(h_lines_uniq)
    h_lines_uniq = [
        LineString([(0, 26), (W, 8)]),
        LineString([(0, 149), (W, 131)]),
        LineString([(0, 273), (W, 250)]),
    ]
    print(h_lines_uniq)
    
    for line in v_lines_uniq + h_lines_uniq:
        (x1, y1), (x2, y2) = line.coords
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
    cv2.imshow('line', img); cv2.waitKey(1)
    
    to_display = img_original.copy()
    cnt = 0
    for h_id in range(len(h_lines_uniq)-1):
        for v_id in range(len(v_lines_uniq)-1):
            left, right = v_lines_uniq[v_id: v_id+2]
            top, bottom = h_lines_uniq[h_id: h_id+2]
            
            left_top = left.intersection(top)
            left_bottom = left.intersection(bottom)
            right_top = right.intersection(top)
            right_bottom = right.intersection(bottom)
            # print(left_top, left_bottom, right_top, right_bottom)
            cv2.circle(img, (int(left_top.x), int(left_top.y)), 5, (0, 255, 0), -1)
            cv2.circle(img, (int(left_bottom.x), int(left_bottom.y)), 5, (0, 255, 0), -1)
            cv2.circle(img, (int(right_top.x), int(right_top.y)), 5, (0, 255, 0), -1)
            cv2.circle(img, (int(right_bottom.x), int(right_bottom.y)), 5, (0, 255, 0), -1)
            # cv2.imshow('line', img); cv2.waitKey(1)
            # cv2.waitKey(0)
            
            # get exp norm
            pts_src = np.array([pt.xy for pt in [left_top, left_bottom, right_bottom, right_top]], dtype=np.float32)
            pts_dst = np.array([[0, 0], [0, 512], [512, 512], [512, 0]], dtype=np.float32)
            h, status = cv2.findHomography(pts_src, pts_dst)
            # print(h, status)
            # assert status == 0
            img_norm = cv2.warpPerspective(img_original, h, (512, 512))
            cv2.imshow('img_norm', img_norm); cv2.waitKey(1)
            
            char_mask = get_mask_single(img_norm)
            cv2.imshow('char_mask', char_mask); cv2.waitKey(1)
            
            score = cv2.matchTemplate(std_mask, char_mask, cv2.TM_CCOEFF_NORMED)[0][0]
            print(score)

            # draw bbox
            contour = np.array([pt.xy for pt in [left_top, left_bottom, right_bottom, right_top]], dtype=np.int32)
            # print(contour)
            cv2.drawContours(to_display, [contour], -1, (0, 255, 0), 2)
            # cv2.imshow('to_display', to_display); cv2.waitKey(0)
            
            # project std to exp
            h_inv = np.linalg.inv(h)
            std_inv = cv2.warpPerspective(std_mask, h_inv, (W, H))
            std_edge = cv2.Canny(std_inv, 50, 100)
            std_contour = cv2.findContours(std_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(to_display, std_contour, -1, (0, 0, 255), 1)
            # cv2.imshow('to_display', to_display); cv2.waitKey(0)

            # put text on that character
            text = f'{score:.2f}'
            cv2.putText(to_display, text, (int(left_top.x), int(left_top.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow('to_display', to_display); cv2.waitKey(1)
            
            cv2.imwrite(f'output/{cnt}.jpg', to_display)
            cnt += 1
    
    cv2.waitKey(0)