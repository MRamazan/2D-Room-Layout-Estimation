import argparse
import random
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import cv2

import numpy
import numpy as np

import os
import sys
import math
sys.path.append("spvloc")
from spvloc_train_test import main_start




def main_process(image_path):


    image = cv2.imread(image_path)
    image = cv2.resize(image, (320, 320))
    os.chdir("../")


    cv2.imwrite(os.path.join("dataset/scene_03457/2D_rendering/141477/perspective/full/0", "rgb_rawlight.png"), image)
    decoded_img_path = "results/results_ps1200_ch1400_ns1_rad1400_h300/03457/000/02b_dec_n.png"
    os.chdir("spvloc")
    main_start()
    os.chdir("../")


    k = encoded_image_scores(decoded_img_path)

    image = color_clustering(decoded_img_path,k)




    shapes = np.unique(image)
    shape_count = len(shapes)
    colors = np.unique(image)
    if len(colors) == 2:
        image[image == colors[0]] = 1  # mid
        image[image == colors[1]] = 50  # left

    if len(colors) == 3:
        image[image == colors[0]] = 1  # mid
        image[image == colors[1]] = 50  # left
        image[image == colors[2]] = 100  # right
    if len(colors) == 4:
        image[image == colors[0]] = 1  # mid
        image[image == colors[1]] = 50  # left
        image[image == colors[2]] = 100  # right
        image[image == colors[3]] = 150  # right
    if len(colors) == 5:
        image[image == colors[0]] = 1  # mid
        image[image == colors[1]] = 50  # left
        image[image == colors[2]] = 100  # right
        image[image == colors[3]] = 150  # left
        image[image == colors[4]] = 200  # right
    img_org = np.copy(image)






    lines = []

    possible_mid_points = calculate_mid_points(image)




    top_border = [[0,0], [319, 0]]
    left_border = [[0, 0], [0, 319]]
    right_border = [[319, 0], [319, 319]]
    bottom_border = [[0, 319], [319, 319]]
    top_border_pixels = image[0,top_border[0][0]:top_border[1][0]]
    left_border_pixels = image[left_border[0][1]:left_border[1][1],0]
    right_border_pixels = image[right_border[0][1]:right_border[1][1],319]
    bottom_border_pixels = image[319,bottom_border[0][0]:bottom_border[1][0]]
    previous_pixel = 1453 #random
    corner_points = []
    for idx,pixel in enumerate(top_border_pixels):
        if pixel != previous_pixel:
            pixel_coord = [idx, 0]
            corner_points.append(pixel_coord)
        previous_pixel = pixel
    previous_pixel = 1453  # random
    for idx,pixel in enumerate(left_border_pixels):
        if pixel != previous_pixel:
            pixel_coord = [0, idx]
            corner_points.append(pixel_coord)
        previous_pixel = pixel
    previous_pixel = 1453  # random
    for idx,pixel in enumerate(right_border_pixels):
        if pixel != previous_pixel:
            pixel_coord = [319, idx]
            corner_points.append(pixel_coord)
        previous_pixel = pixel
    previous_pixel = 1453
    for idx,pixel in enumerate(bottom_border_pixels):
        if pixel != previous_pixel:
            pixel_coord = [idx, 319]
            corner_points.append(pixel_coord)
        previous_pixel = pixel

    border_top_corners = []
    border_left_corners = []
    border_right_corners = []
    border_bottom_corners = []
    for point in corner_points:
        x,y = point[0],point[1]
        if point != [0, 0] and point != [0, 319] and point != [319, 0] and point != [319,319]:
         if x == 0 and y in range(0,319):
            border_left_corners.append(point)
         if x == 319 and y in range(0,319):
            border_right_corners.append(point)
         if x in range(0,319) and y == 0:
            border_top_corners.append(point)
         if x in range(0,319) and y == 319:
            border_bottom_corners.append(point)

    max_top_distance = 1000  # random
    top_nearest = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    max_left_distance = 1000 # random
    left_nearest = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    max_right_distance = 1000  # random
    right_nearest = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    max_bot_distance = 1000  # random
    bot_nearest = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]




    for idx,corner in enumerate(border_left_corners):
        max_left_distance = 1000
        for point in possible_mid_points:
           left_distance = math.sqrt((point[0] - border_left_corners[idx][0])**2 + (point[1] - border_left_corners[idx][1])**2)

           if left_distance < max_left_distance:
              max_left_distance = left_distance
              left_nearest[idx] = point

    for idx, corner in enumerate(border_right_corners):
        max_right_distance = 1000
        for point in possible_mid_points:
            right_distance = math.sqrt(
                (point[0] - border_right_corners[idx][0]) ** 2 + (point[1] - border_right_corners[idx][1]) ** 2)

            if right_distance < max_right_distance:
                max_right_distance = right_distance
                print(idx)
                right_nearest[idx] = point
    for idx, corner in enumerate(border_top_corners):
        skip = False
        for point in possible_mid_points:
            top_distance = math.sqrt(
                (point[0] - border_top_corners[idx][0]) ** 2 + (point[1] - border_top_corners[idx][1]) ** 2)


            if top_distance < 30:
                top_nearest[idx] = point
                skip = True
                break

        if not skip:
          max_top_distance = 1000
          for point in possible_mid_points:
            top_distance = math.sqrt(
                (point[0] - border_top_corners[idx][0]) ** 2 + (point[1] - border_top_corners[idx][1]) ** 2)


            if abs(border_top_corners[idx][0] - point[0]) < 15:
                top_nearest[idx] = point

                break
            if top_distance < max_top_distance:
                max_top_distance = top_distance
                top_nearest[idx] = point



    for idx, corner in enumerate(border_bottom_corners):
        max_bot_distance = 1000
        for point in possible_mid_points:
            bottom_distance = math.sqrt(
                (point[0] - border_bottom_corners[idx][0]) ** 2 + (point[1] - border_bottom_corners[idx][1]) ** 2)

            if bottom_distance < max_bot_distance:
                max_bot_distance = bottom_distance
                bot_nearest[idx] = point


    for idx, point in enumerate(top_nearest):
        if len(point) > 0:
            cv2.line(image, (border_top_corners[idx][0], border_top_corners[idx][1]), (point[0], point[1]),
                     color=(255), thickness=2)
            lines.append([border_top_corners[idx],point])


    for idx, point in enumerate(left_nearest):
        if len(point) > 0:
            cv2.line(image, (border_left_corners[idx][0], border_left_corners[idx][1]), (point[0], point[1]),
                     color=(255), thickness=2)
            lines.append([border_left_corners[idx], point])
    for idx, point in enumerate(right_nearest):
        if len(point) > 0:
            cv2.line(image, (border_right_corners[idx][0], border_right_corners[idx][1]), (point[0], point[1]),
                     color=(255), thickness=2)
            lines.append([border_right_corners[idx], point])

    for idx, point in enumerate(bot_nearest):
        if len(point) > 0:
            cv2.line(image, (border_bottom_corners[idx][0], border_bottom_corners[idx][1]), (point[0], point[1]),
                     color=(255), thickness=2)
            lines.append([border_bottom_corners[idx], point])

    x_coords = np.array(possible_mid_points)[:,0]
    y_coords = np.array(possible_mid_points)[:,1]
    for coord in x_coords:
        for coord2 in x_coords:
            if coord == coord2:
                possible_mid_points[list(x_coords).index(coord)][0] = coord + 1
                break

    for coord in y_coords:
        for coord2 in y_coords:
            if coord == coord2:
                possible_mid_points[list(y_coords).index(coord)][1] = coord + 1
                break



    if shape_count == 4:
        left = True
        right = True
        roof = True
        ceiling = True
        mid = False
        lines_to_assemble_ground = []
        lines_to_assemble_ceiling = []
        lines_to_assemble_left = []
        lines_to_assemble_right = []

        top_mid_point = np.min(np.array(possible_mid_points)[:,1])
        bottom_mid_point = np.max(np.array(possible_mid_points)[:,1])
        left_line = [[1453,1453],[1881,1938]]#random
        mid_to_top = []
        mid_to_bot = []
        for line in lines:
            if np.any(line == bottom_mid_point):
                mid_to_bot.append(line)
                lines_to_assemble_ground.append(line)

                if line[0][0] < left_line[0][0]:
                    left_line = line
            elif np.any(line == top_mid_point):
                mid_to_top.append(line)

        lines_to_assemble_left.append(left_line)

        left_corner_ground = [1453,1453]#random
        right_corner_ground = [1453,1453]

        left_corner_top = [1453, 1453]  # random
        right_corner_top = [1453, 1453]

        left_line_top = None

        in_left_border = False
        in_right_border = False
        left_in_bottom = False
        right_in_bottom = False
        both_bottom_border = False

        in_left_border2 = False
        in_right_border2 = False
        left_in_top = False
        right_in_top = False
        both_top_border = False

        for line in mid_to_bot:
            if line[0][0] < left_corner_ground[0]:
              left_corner_ground = line[0]

        for line in mid_to_top:
            if line[0][0] < left_corner_top[0]:
              left_corner_top = line[0]
              left_line_top = line


        for line in mid_to_bot:
            if line[0][0] > left_corner_ground[0]:
              right_corner_ground = line[0]


        for line in mid_to_top:
            if line[0][0] > left_corner_top[0]:
              right_corner_top = line[0]


        lines_to_assemble_left.append(left_line_top)


        for point in border_left_corners:
            if np.array_equal(point, left_corner_ground):
                in_left_border = True


        for point in border_left_corners:
            if np.array_equal(point, left_corner_top):
                in_left_border2 = True


        for point in border_right_corners:
            if np.array_equal(point, right_corner_ground):
                in_right_border = True

        for point in border_right_corners:
            if np.array_equal(point, right_corner_top):
                in_right_border2 = True


        for point in border_bottom_corners:
            if np.array_equal(point, left_corner_ground):
                left_in_bottom = True
            for point2 in border_bottom_corners:
                if np.array_equal(point2, right_corner_ground):
                    right_in_bottom = True
            if left_in_bottom and right_in_bottom:
                both_bottom_border = True


        for point in border_top_corners:
            if np.array_equal(point, left_corner_top):
                left_in_top = True
            for point2 in border_top_corners:
                if np.array_equal(point2, right_corner_top):
                    right_in_top = True
            if left_in_top and right_in_top:
                both_top_border = True

        if both_bottom_border:
            lines_to_assemble_ground.append([left_corner_ground, right_corner_ground])
        if in_left_border and in_right_border:
            lines_to_assemble_ground.append([[0,319], [319,319]])
            lines_to_assemble_ground.append([left_corner_ground, [0, 319]])
            lines_to_assemble_ground.append([right_corner_ground, [319, 319]])
        if in_left_border and right_in_bottom:
            lines_to_assemble_ground.append([[0,319], right_corner_ground])
        if in_right_border and left_in_bottom:
            lines_to_assemble_ground.append([[319, 319], left_corner_ground])
            lines_to_assemble_ground.append([[319, 319], right_corner_ground])
        if in_left_border and in_left_border2:
            lines_to_assemble_left.append([left_corner_ground, left_corner_top])
        if in_left_border and left_in_top:
            lines_to_assemble_left.append([left_corner_ground, [0,0]])
            lines_to_assemble_left.append([[0,0], left_corner_top])
        if in_left_border2 and left_in_bottom:
            lines_to_assemble_left.append([left_corner_top, [0,319]])
            lines_to_assemble_left.append([[0,319], left_corner_ground])
        if left_in_bottom and left_in_top:

            lines_to_assemble_left.append([left_corner_top, [0,0]])
            lines_to_assemble_left.append([[0,0], [0, 319]])
            lines_to_assemble_left.append([[0,319], left_corner_ground])
        if len(possible_mid_points) == 2:
            lines_to_assemble_left.append([possible_mid_points[0], possible_mid_points[1]])



    if len(possible_mid_points) == 2:
        cv2.line(image, (possible_mid_points[0][0], possible_mid_points[0][1]),
                 (possible_mid_points[1][0], possible_mid_points[1][1]), color=(255, 255, 255), thickness=2)
        lines.append([possible_mid_points[0],possible_mid_points[1]])
    if len(possible_mid_points) == 4:
        possible_mid_points = np.array(possible_mid_points)

        x_cords = possible_mid_points[:, 0]
        sorted_numbers = sorted(x_cords)
        smallest_two = sorted_numbers[:2]
        biggest_two = sorted_numbers[2:]
        top_left_and_bot_left_idxs = list(x_cords).index(smallest_two[0]), list(x_cords).index(smallest_two[1])
        top_right_and_bot_right_idxs = list(x_cords).index(biggest_two[0]), list(x_cords).index(biggest_two[1])
        left_corner1 = list(possible_mid_points[top_left_and_bot_left_idxs[0]])
        left_corner2 = list(possible_mid_points[top_left_and_bot_left_idxs[1]])
        right_corner1 = list(possible_mid_points[top_right_and_bot_right_idxs[0]])
        right_corner2 = list(possible_mid_points[top_right_and_bot_right_idxs[1]])
        if abs(left_corner1[1] - right_corner1[1]) > abs(left_corner1[1] - right_corner2[1]):
            cv2.line(image, (left_corner1[0], left_corner1[1]), (right_corner2[0], right_corner2[1]),
                     color=(255), thickness=2)
            lines.append([left_corner1, right_corner2])
        else:
            cv2.line(image, (left_corner1[0], left_corner1[1]), (right_corner1[0], right_corner1[1]),
                     color=(255), thickness=2)
            lines.append([left_corner1, right_corner1])
        if abs(left_corner2[1] - right_corner1[1]) > abs(left_corner2[1] - right_corner2[1]):
            cv2.line(image, (left_corner2[0], left_corner2[1]), (right_corner2[0], right_corner2[1]),
                     color=(255), thickness=2)
            lines.append([left_corner2, right_corner2])
        else:
            cv2.line(image, (left_corner2[0], left_corner2[1]), (right_corner1[0], right_corner1[1]),
                     color=(255), thickness=2)
            lines.append([left_corner2, right_corner1])

        cv2.line(image, (left_corner1[0], left_corner1[1]), (left_corner2[0], left_corner2[1]), color=(255),
                 thickness=2)
        lines.append([left_corner1, left_corner2])
        cv2.line(image, (right_corner1[0], right_corner1[1]), (right_corner2[0], right_corner2[1]),
                 color=(255), thickness=2)
        lines.append([right_corner1,right_corner2])
    if len(possible_mid_points) > 4:
        possible_mid_points = np.array(possible_mid_points)
        x_coords = possible_mid_points[:, 0]
        for coord in x_coords:
            for coord2 in x_coords:
                if abs(coord - coord2) < 15:
                    corner1_idx = list(x_coords).index(coord)
                    corner2_idx = list(x_coords).index(coord2)
                    corner1 = possible_mid_points[corner1_idx]
                    corner2 = possible_mid_points[corner2_idx]
                    cv2.line(image, (corner1[0], corner1[1]), (corner2[0], corner2[1]), color=(255), thickness=2)
                    lines.append([corner1, corner2])
        for point in possible_mid_points:
            min_distance = 1453
            nearest = 0
            for point2 in possible_mid_points:
                distance = calculate_distance(point, point2)
                if list(point) != list(point2):
                 if distance < min_distance:
                    min_distance = distance
                    nearest = point2

            cv2.line(image, (point[0], point[1]), (nearest[0], nearest[1]), color=(255), thickness=2)
            lines.append([point, nearest])

    if len(possible_mid_points) == 3:
        mid_point_opt_connection = 3
        possible_mid_points = np.array(possible_mid_points)
        x_coords = possible_mid_points[:, 0]
        y_coords = possible_mid_points[:, 1]
        mid_connections = []

        for coord in x_coords:
            for coord2 in x_coords:
                if abs(coord - coord2) < 15:
                    corner1_idx = list(x_coords).index(coord)
                    corner2_idx = list(x_coords).index(coord2)
                    corner1 = possible_mid_points[corner1_idx]
                    corner2 = possible_mid_points[corner2_idx]
                    cv2.line(image, (corner1[0], corner1[1]), (corner2[0], corner2[1]), color=(255), thickness=2)
                    mid_connections.append([corner1, corner2])
                    lines.append([corner1, corner2])

        for coord in y_coords:
            for coord2 in y_coords:
                if abs(coord - coord2) < 15:
                    corner1_idx = list(y_coords).index(coord)
                    corner2_idx = list(y_coords).index(coord2)
                    corner1 = possible_mid_points[corner1_idx]
                    corner2 = possible_mid_points[corner2_idx]
                    cv2.line(image, (corner1[0], corner1[1]), (corner2[0], corner2[1]), color=(255), thickness=2)
                    mid_connections.append([list(corner1), list(corner2)])
                    lines.append([corner1, corner2])

        del_idxs = []


        for index  in range(len(mid_connections)):
            line1 = mid_connections[index][0]
            line2 = mid_connections[index][1]
            if np.array_equal(line1, line2):
                del_idxs.append(index)


        for num,idx in enumerate(del_idxs):
            if num == 0:
                del mid_connections[idx - num]
            else:
                del mid_connections[idx - num]

        lines = convert_numpy_to_list(lines)

        deleted = 0
        for idx, conn in enumerate(mid_connections):
           if is_equal(list(mid_connections[idx - deleted]), list(mid_connections[idx - deleted])):
              del mid_connections[idx - deleted]
              deleted += 1

        for iter in range(0,10):
          for line in lines:
            if np.array_equal(np.array(line[0]), np.array(line[1])):
                lines.remove(line)

        for line in lines:
            for line2 in lines:
             if not np.array_equal(np.array(line), np.array(line2)):
              if is_equal(line, line2):
                lines.remove(line)
              else:
                 continue

        mid_conns = {}

        for point in possible_mid_points:
            key = create_key(point)
            mid_conns[key] = []
            for line in lines:
                if np.array_equal(point, np.array(line[0])) or np.array_equal(point, np.array(line[1])):
                    mid_conns[key].append(line)

        mid3_2 = False
        count_2 = 0
        pts = []




        for key,value in mid_conns.items():
            if len(value) == 3:
                continue
            if len(value) == 2:
                count_2 += 1
                pts.append(key)

        if count_2 == 2:
            mid3_2 = True
            if mid3_2:
                print(pts)

                pts = [eval(s.replace("  ", " ").replace(" ", ", ")) for s in pts]
                print(pts)
                cv2.line(image, (pts[0]), (pts[1]),color=(255), thickness=2)
                lines.append(pts)

        conn_count = 0
        for point in possible_mid_points:
           for line in lines:
               if is_in(point, line):

                   conn_count += 1




    if len(border_left_corners) == 0:
        cv2.line(image, (0, 0), (0, 319), color=(255), thickness=2)
        lines.append([[0, 0], [0, 319]])
    for point in border_left_corners:
        if len(border_left_corners) >= 2:

            distances = {}
            distance1 = calculate_distance(point, [0, 0])
            distance2 = calculate_distance(point, [0, 319])
            distances["[0, 0]"] = distance1
            distances["[0, 319]"] = distance2
            nearest_to_0 = False
            nearest_to_319 = False
            for point2 in border_left_corners:
                if point2 != point:
                    distances[create_key(point2)] = calculate_distance(point, point2)

            if max([list[1] for list in border_left_corners]) == point[1]:
                nearest_to_319 = True
            if min([list[1] for list in border_left_corners]) == point[1]:
                nearest_to_0 = True
            if nearest_to_319:
                nearest_two_key = sorted(distances.keys(), key=distances.get)[:2]
                converted_two_key = [eval(item) for item in nearest_two_key]
                if converted_two_key[0] == [0, 319]:
                    nearest_key = converted_two_key[1]
                else:
                    nearest_key = converted_two_key[0]

                cv2.line(image, (point), ([0, 319]), color=(255), thickness=2)
                lines.append([point, [0, 319]])
                cv2.line(image, (point), (nearest_key), color=(255), thickness=2)
                lines.append([point, nearest_key])

                continue
            elif nearest_to_0:
                nearest_two_key = sorted(distances.keys(), key=distances.get)[:2]
                converted_two_key = [eval(item) for item in nearest_two_key]
                if converted_two_key[0] == [0, 0]:
                    nearest_key = converted_two_key[1]
                else:
                    nearest_key = converted_two_key[0]

                cv2.line(image, (point), ([0, 0]), color=(255), thickness=2)
                lines.append([point, [0, 0]])
                cv2.line(image, (point), (nearest_key), color=(255), thickness=2)
                lines.append([point, nearest_key])

                continue
            else:

                nearest_two_key = sorted(distances.keys(), key=distances.get)[:2]
                converted_two_key = [eval(item) for item in nearest_two_key]

                cv2.line(image, (point), (converted_two_key[0]), color=(255), thickness=2)
                lines.append([point, converted_two_key[0]])
                cv2.line(image, (point), (converted_two_key[1]), color=(255), thickness=2)
                lines.append([point, converted_two_key[1]])


        elif len(border_left_corners) == 1:
            cv2.line(image, (point), ([0, 0]), color=(255), thickness=2)
            cv2.line(image, (point), ([0, 319]), color=(255), thickness=2)
            lines.append([point, [0, 0]])
            lines.append([point, [0, 319]])
    if len(border_right_corners) == 0:
        cv2.line(image, ([319,0]), ([319,319]), color=(255), thickness=2)
        lines.append([[319,0], [319,319]])

    for point in border_right_corners:
        if len(border_right_corners) >= 2:

            distances = {}
            distance1 = calculate_distance(point, [319, 0])
            distance2 = calculate_distance(point, [319, 319])
            distances["[319, 0]"] = distance1
            distances["[319, 319]"] = distance2
            nearest_to_319 = False
            nearest_to_d319 = False
            for point2 in border_right_corners:
                if point2 != point:
                    distances[create_key(point2)] = calculate_distance(point, point2)

            print(max([list[1] for list in border_right_corners]), "yarak")
            if min([list[1] for list in border_right_corners]) == point[1]:
                nearest_to_319 = True
            if max([list[1] for list in border_right_corners]) == point[1]:
                nearest_to_d319 = True
            if nearest_to_319:
                nearest_two_key = sorted(distances.keys(), key=distances.get)[:2]
                converted_two_key = [eval(item) for item in nearest_two_key]
                if converted_two_key[0] == [319, 0]:
                    nearest_key = converted_two_key[1]
                else:
                    nearest_key = converted_two_key[0]

                cv2.line(image, (point), ([319, 0]), color=(255), thickness=2)
                lines.append([point, [319, 0]])
                cv2.line(image, (point), (nearest_key), color=(255), thickness=2)
                lines.append([point, nearest_key])

                continue
            elif nearest_to_d319:
                nearest_two_key = sorted(distances.keys(), key=distances.get)[:2]
                converted_two_key = [eval(item) for item in nearest_two_key]
                if converted_two_key[0] == [319, 319]:
                    nearest_key = converted_two_key[1]
                else:
                    nearest_key = converted_two_key[0]

                cv2.line(image, (point), ([319, 319]), color=(255), thickness=2)
                lines.append([point, [319, 319]])
                cv2.line(image, (point), (nearest_key), color=(255), thickness=2)
                lines.append([point, nearest_key])

                continue
            else:

                nearest_two_key = sorted(distances.keys(), key=distances.get)[:2]
                converted_two_key = [eval(item) for item in nearest_two_key]

                cv2.line(image, (point), (converted_two_key[0]), color=(255), thickness=2)
                lines.append([point, converted_two_key[0]])
                cv2.line(image, (point), (converted_two_key[1]), color=(255), thickness=2)
                lines.append([point, converted_two_key[1]])


        elif len(border_right_corners) == 1:
            cv2.line(image, (point), ([319, 319]), color=(255), thickness=2)
            cv2.line(image, (point), ([319, 0]), color=(255), thickness=2)
            lines.append([point, [319, 319]])
            lines.append([point, [319, 0]])

    if len(border_bottom_corners) == 0:
        cv2.line(image, (0, 319), (319, 319), color=(255), thickness=2)
        lines.append([[0, 319], [319, 319]])
    for point in border_bottom_corners:
        if len(border_bottom_corners) >= 2:

            distances = {}
            distance1 = calculate_distance(point, [0, 319])
            distance2 = calculate_distance(point, [319, 319])
            distances["[0, 319]"] = distance1
            distances["[319, 319]"] = distance2
            nearest_to_319 = False
            nearest_to_0 = False
            for point2 in border_bottom_corners:
                if point2 != point:
                    distances[create_key(point2)] = calculate_distance(point, point2)

            if max([list[0] for list in border_bottom_corners]) == point[0]:
                nearest_to_319 = True
            if min([list[0] for list in border_bottom_corners]) == point[0]:
                nearest_to_0 = True
            if nearest_to_319:
                nearest_two_key = sorted(distances.keys(), key=distances.get)[:2]
                converted_two_key = [eval(item) for item in nearest_two_key]
                if converted_two_key[0] == [319, 319]:
                    nearest_key = converted_two_key[1]
                else:
                    nearest_key = converted_two_key[0]

                cv2.line(image, (point), ([319, 319]), color=(255), thickness=2)
                lines.append([point, [319, 319]])
                cv2.line(image, (point), (nearest_key), color=(255), thickness=2)
                lines.append([point, nearest_key])

                continue
            elif nearest_to_0:
                nearest_two_key = sorted(distances.keys(), key=distances.get)[:2]
                converted_two_key = [eval(item) for item in nearest_two_key]
                if converted_two_key[0] == [0, 319]:
                    nearest_key = converted_two_key[1]
                else:
                    nearest_key = converted_two_key[0]

                cv2.line(image, (point), ([0, 319]), color=(255), thickness=2)
                lines.append([point, [0, 319]])
                cv2.line(image, (point), (nearest_key), color=(255), thickness=2)
                lines.append([point, nearest_key])

                continue
            else:

                nearest_two_key = sorted(distances.keys(), key=distances.get)[:2]
                converted_two_key = [eval(item) for item in nearest_two_key]

                cv2.line(image, (point), (converted_two_key[0]), color=(255), thickness=2)
                lines.append([point, converted_two_key[0]])
                cv2.line(image, (point), (converted_two_key[1]), color=(255), thickness=2)
                lines.append([point, converted_two_key[1]])


        elif len(border_bottom_corners) == 1:
            cv2.line(image, (point), ([0, 319]), color=(255), thickness=2)
            cv2.line(image, (point), ([319, 319]), color=(255), thickness=2)
            lines.append([point, [0, 319]])
            lines.append([point, [319, 319]])
    else:
     for point in border_top_corners:
        if len(border_top_corners) >= 2:

            distances = {}
            distance1 = calculate_distance(point, [319, 0])
            distance2 = calculate_distance(point, [0, 0])
            distances["[319, 0]"] = distance1
            distances["[0, 0]"] = distance2
            nearest_to_319 = False
            nearest_to_0 = False
            for point2 in border_top_corners:
                if point2 != point:
                    distances[create_key(point2)] = calculate_distance(point, point2)

            if max([list[0] for list in border_top_corners]) == point[0]:
                nearest_to_319 = True
            if min([list[0] for list in border_top_corners]) == point[0]:
                nearest_to_0 = True
            if nearest_to_319:
                nearest_two_key = sorted(distances.keys(), key=distances.get)[:2]
                converted_two_key = [eval(item) for item in nearest_two_key]
                if converted_two_key[0] == [319, 0]:
                    nearest_key = converted_two_key[1]
                else:
                    nearest_key = converted_two_key[0]

                cv2.line(image, (point), ([319,0]), color=(255), thickness=2)
                lines.append([point, [319, 0]])
                cv2.line(image, (point), (nearest_key), color=(255), thickness=2)
                lines.append([point, nearest_key])

                continue
            elif nearest_to_0:
                nearest_two_key = sorted(distances.keys(), key=distances.get)[:2]
                converted_two_key = [eval(item) for item in nearest_two_key]
                if converted_two_key[0] == [0, 0]:
                    nearest_key = converted_two_key[1]
                else:
                    nearest_key = converted_two_key[0]

                cv2.line(image, (point), ([0, 0]), color=(255), thickness=2)
                lines.append([point, [0, 0]])
                cv2.line(image, (point), (nearest_key), color=(255), thickness=2)
                lines.append([point, nearest_key])

                continue
            else:


                nearest_two_key = sorted(distances.keys(), key=distances.get)[:2]
                converted_two_key = [eval(item) for item in nearest_two_key]


                cv2.line(image, (point), (converted_two_key[0]), color=(255), thickness=2)
                lines.append([point, converted_two_key[0]])
                cv2.line(image, (point), (converted_two_key[1]), color=(255), thickness=2)
                lines.append([point, converted_two_key[1]])


        elif len(border_top_corners) == 1:
            cv2.line(image, (point), ([0,0]), color=(255), thickness=2)
            cv2.line(image, (point), ([319, 0]), color=(255), thickness=2)
            lines.append([point, [0,0]])
            lines.append([point, [319, 0]])
    new_lines  =[]
    for line in lines:
        for line2 in line:
         if isinstance(line2, numpy.ndarray):
            line = [arr.tolist() for arr in line]
            break
        new_lines.append(line)



    lines = convert_numpy_to_list(lines)

    for line in lines:
        cv2.line(image,(line[0][0], line[0][1]), (line[1][0],line[1][1]), color=(255), thickness=2)



    for iter in range(0, 10):
        for line in lines:
            if np.array_equal(np.array(line[0]), np.array(line[1])):

                lines.remove(line)


    for line in lines:
        for line2 in lines:
            if not np.array_equal(np.array(line), np.array(line2)):
                if is_equal(line, line2):
                    lines.remove(line)
                else:
                    continue

    image1 = Image.open(image_path).resize((320,320))
    image2 = Image.fromarray(image)


    image1.paste(image2, (0, 0), image2)

    image1.show()



def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def calculate_distance(p1, p2):
    distance = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return distance


def calculate_mid_points(image):
    possible_mid_points = []
    for column in range(0,320):
        for row in range(0, 320):

           square_size = 2
           half_square = square_size // 2
           square = image[int(row) - half_square:int(row) + half_square + 1,
                 int(column) - half_square:int(column) + half_square + 1]

           color_num = np.unique(square)
           if len(color_num) == 3:
               possible_mid_points.append([int(column), int(row)])
    same_corners = []
    mid_corners = []

    for point in possible_mid_points:
        for point2 in possible_mid_points:
            if point != point2:
                distance = calculate_distance(point, point2)
                if distance < 10:
                   same_corners.append(point2)
        if len(same_corners) > 0:
          rand_idx = random.randint(0,len(same_corners) - 1)
          mid_corners.append(same_corners[rand_idx])
        else:
          mid_corners.append(point)
        for corner in same_corners:
            possible_mid_points.remove(corner)
        same_corners = []


    return mid_corners



def optimal_number_of_clusters(wcss, silhouette_scores, cluster_range):

    wcss_differences = np.diff(wcss)

    elbow_point = np.argmax(wcss_differences) + 2  # +2 çünkü np.diff uzunluğu bir eksik olur


    best_silhouette_k = cluster_range[np.argmax(silhouette_scores)]

    if abs(best_silhouette_k - elbow_point) <= 1:
        return elbow_point
    else:
        return best_silhouette_k

def encoded_image_scores(image_path):
    image = cv2.imread(image_path)
    image = image[59:202, 0:319]
    image = cv2.resize(image,(320,320))


    image = Image.fromarray(image)

    image = image.convert('RGB')

    image_array = np.array(image)

    pixels = image_array.reshape(-1, 3)

    scaler = StandardScaler()
    pixels_normalized = scaler.fit_transform(pixels)

    pixels_normalized_sample = shuffle(pixels_normalized, random_state=42)[:1000]

    k_values = range(2, 6)


    wcss = []
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(pixels_normalized)

        wcss.append(kmeans.inertia_)

        silhouette_sample = kmeans.predict(pixels_normalized_sample)
        silhouette_scores.append(silhouette_score(pixels_normalized_sample, silhouette_sample))
    optimal_k = optimal_number_of_clusters(wcss, silhouette_scores, list(k_values))

    return optimal_k
def is_equal(list1, list2):
    list1 = convert_numpy_to_list(list1)
    list2 = convert_numpy_to_list(list2)
    if  list1[0] in list2 and list1[1] in list2:
        return  True
    else:
        return False

def is_in(list1, list2):
    for list in list2:
        if np.array_equal(np.array(list1), np.array(list)):
            return True
    return False
def convert_numpy_to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return [convert_numpy_to_list(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_numpy_to_list(value) for key, value in data.items()}
    elif isinstance(data, tuple):
        return tuple(convert_numpy_to_list(item) for item in data)
    else:
        return data

def color_clustering(image_path,k):
    image = cv2.imread(image_path)
    image = image[59:202, 0:319]
    image = cv2.resize(image, (320,320))



    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pixel_values = image_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)



    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image_rgb.shape)


    gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)

    return gray_segmented

def create_key(*args):
    return "_".join(map(str, args))

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Layout Project Parser')
    parse.add_argument('--image_path', type=str,help='path to image file',default="")
    args = parse.parse_args()

    main_process(args.image_path)
