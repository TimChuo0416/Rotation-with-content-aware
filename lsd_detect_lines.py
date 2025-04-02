import cv2
import numpy as np
from lsd.lsd import lsd


def detect_lines(img_array, angle, x_len, y_len, x, y, minLen=2, M=90):
    src = img_array.copy()
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    lines = lsd(gray)
    legal_lines = list()
    bin_count = np.zeros((90))
    for i in range(lines.shape[0]):
        pt1 = [int(lines[i, 0]), int(lines[i, 1])]
        pt2 = [int(lines[i, 2]), int(lines[i, 3])]
        vec = [pt1[0] - pt2[0], pt1[1] - pt2[1]]
        if np.linalg.norm(np.array(vec)) >= minLen:
            temp_lines = [pt1, pt2]

            # used for calculating the intersection of line and mesh_grids
            left_x, right_x = min(pt1[0], pt2[0]), max(pt1[0], pt2[0])
            top_y, bottom_y = min(pt1[1], pt2[1]), max(pt1[1], pt2[1])

            # x_mesh_left >= left_x, x_mesh_right <= right_x
            x_mesh_left = np.ceil(left_x / x_len) * x_len
            x_mesh_right = np.floor(right_x / x_len) * x_len

            # y_mesh_bottom >= bottom_y, y_mesh_top <= top_y
            y_mesh_top = np.ceil(top_y / y_len) * y_len
            y_mesh_bottom = np.floor(bottom_y / y_len) * y_len

            # record the intersection coordinate of line and x-axis-mesh-grid
            while x_mesh_left <= x_mesh_right:
                mesh_start, mesh_end = [x_mesh_left, 0], [x_mesh_left, y]
                flag, p = intersect(pt1, pt2, mesh_start, mesh_end)
                if flag:
                    temp_lines.append(p)
                x_mesh_left += x_len

            # record the intersection coordinate of line and y-axis-mesh-grid
            while y_mesh_top <= y_mesh_bottom:
                mesh_start, mesh_end = [0, y_mesh_top], [x, y_mesh_top]
                flag, p = intersect(pt1, pt2, mesh_start, mesh_end)
                if flag:
                    temp_lines.append(p)
                y_mesh_top += y_len

            # sort temp_lines in y_coordinate to get sub_lines
            temp_lines.sort(key=lambda coordinates: coordinates[1])
            # bin_count = np.zeros((90))
            for j in range(len(temp_lines) - 1):
                l_start, l_end = temp_lines[j], temp_lines[j + 1]
                l_vec = [l_start[0] - l_end[0], l_start[1] - l_end[1]]
                if np.linalg.norm(np.array(l_vec)) > minLen:
                    # calculate  the orientation between line and x-axis, [0, 180)
                    theta = np.arctan(l_vec[1] / (l_vec[0] + 1e-10)) / np.pi * 180.0
                    theta = theta if theta >= 0 else theta + 180.0
                    bin_num = int(np.ceil(theta + angle) / 180.0 * M)

                    # bin_num = [1, 90]
                    if bin_num <= 0:
                        bin_num += 90
                    elif bin_num > 90:
                        bin_num -= 90
                    legal_lines.append([l_start[0], l_start[1], l_end[0], l_end[1], theta, bin_num])
                    bin_count[bin_num-1] = bin_count[bin_num-1] + 1

    line_check = np.zeros((90))
    lines_in_bins = np.empty((90, ), dtype=object)
    for i in range(90):
        lines_in_bins[i] = []
    
    for line in legal_lines:
        lines_in_bins[line[5]-1].append(line)
        line_check[line[5]-1] = line_check[line[5]-1]+1
        # print(f"bin num:{line[5]}\n")

    for i in range(90):
        # print(f"bin count:{bin_count[i]}, line check:{line_check[i]}")
        if bin_count[i] != line_check[i]:
            print(f"{i} is off by {bin_count - line_check[i]}")

    # for line in lines_in_bins:
    #     print(f"{len(line)}")
    merged_lines = merge_hough_lines(lines_in_bins, distance_threshold=5)
    # return legal_lines
    return merged_lines

def merge_hough_lines(lines, distance_threshold=5):
    """
    Merge lines based on their angle and proximity.
    
    :param lines: List of lines as [x1, y1, x2, y2].
    :param angle_threshold: Maximum angular difference to consider lines similar.
    :param distance_threshold: Maximum distance between lines to consider merging.
    :return: List of merged lines.
    """
    # print(f"{lines[0]}")
    final_lines = list()
    
    
    def distance_between_lines(line1, line2):
        # Calculate the distance between two lines (midpoints)
        x1, y1, x2, y2 = line1[0], line1[1], line1[2], line1[3]
        x3, y3, x4, y4 = line2[0], line2[1], line2[2], line2[3]
        mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        mid2 = ((x3 + x4) / 2, (y3 + y4) / 2)
        return np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)

    

    for num, bin_list in enumerate(lines):
        merged_lines = []
        used = set()
        # print(f"nums:{num}, elements:{bin_list}")
        for i, line1 in enumerate(bin_list):
            if i in used:
                continue
            merged = line1
            for j, line2 in enumerate(bin_list):
                if j in used or i == j:
                    continue
                if distance_between_lines(line1, line2) < distance_threshold:
                    # Merge lines by extending to the farthest endpoints
                    merged[0], merged[1] = min(merged[0], line2[0]), min(merged[1], line2[1])
                    merged[2], merged[3] = max(merged[2], line2[2]), max(merged[3], line2[3])
                    used.add(j)
            used.add(i)
            # merged_lines.append(merged)
            merged_lines.append([merged[0], merged[1], merged[2], merged[3], (line1[4]+line2[4])/2, num])
            final_lines.append([merged[0], merged[1], merged[2], merged[3], (line1[4]+line2[4])/2, num])

    # # return merged_lines
    return final_lines


def intersect(pt1, pt2, pt3, pt4):
    # return the intersection of pt1 - pt2 and pt3 - pt4
    a0 = pt1[1] - pt2[1]
    b0 = pt2[0] - pt1[0]
    c0 = pt1[0] * pt2[1] - pt2[0] * pt1[1]

    a1 = pt3[1] - pt4[1]
    b1 = pt4[0] - pt3[0]
    c1 = pt3[0] * pt4[1] - pt4[0] * pt3[1]

    d = a0 * b1 - a1 * b0
    if d == 0:
        return False, None
    x = (b0 * c1 - b1 * c0) / d
    y = (a1 * c0 - a0 * c1) / d
    return True, [x, y]
