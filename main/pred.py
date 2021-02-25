#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 9 23:11:51 2020
table line detect
@author: chineseocr
"""

import time
import os
import cv2
import json
from skimage import measure, color
from utils.ocr_utils import nparray2base64
from util.line_util import filter_lines, sort_rowlines, row_line_merge, sort_collines
from utils.utils import draw_lines, minAreaRectBox, get_table_cellboxes
from main.table_line import table_line
import logging

logger = logging.getLogger("__name__")


def get_cell(h, w, img, colboxes, rowboxes):
    from util.line_to_cell import VertexDetect, CellDetect, filter_vertex, cell_extert
    # 求交点
    cross_points, image = VertexDetect(img, colboxes, rowboxes)
    # print("cross_points:",cross_points)
    #cv2.imwrite("test/points2.jpg", image)
    # 过滤x、y坐标
    mylistx, mylisty = filter_vertex(cross_points)
    # 单元格提取，返回行列索引
    row_col_index_and_pts, img = cell_extert(h, w, img, mylistx, mylisty)

    #cv2.imwrite(os.path.join("/Users/yanmeima/Desktop/" + "F2012201923500180118+保单缺失_6.jpg"), img)


def main(input_path, output_path):
    img = cv2.imread(input_path)
    h, w, _ = img.shape
    print(img.shape)
    # 微调
    from util.table_util import RotateProcessor
    rotate_processor = RotateProcessor()
    tuning_degree, tuning_image = rotate_processor.process(img)
    img = tuning_image
    rowboxes, colboxes = table_line(img[..., ::-1], size=(1024, 1024), hprob=0.5, vprob=0.5)

    draw_img = draw_lines(img, rowboxes + colboxes, color=(255, 0, 0), lineW=2)
    # print(time.time() - t, len(rowboxes), len(colboxes))
    # cv2.imwrite("data/line.jpg", draw_img)

    from util import table_detect
    tables = table_detect.table_detect(img, rowboxes, colboxes)

    # 5、做透射变换，所有坐标都透射
    boxes = []
    # TODO
    for table in tables:
        table_pos = table['vertex']
        row_lines = table['rowlines']
        col_lines = table['collines']
        # todo
        w, h, new_pos, row_lines = table_detect.perspective(table_pos, row_lines)
        w, h, new_pos, col_lines = table_detect.perspective(table_pos, col_lines)
        # 暴力上下左右打通的测试
        cells, row_num, col_num = get_cell(h, w, img, row_lines, col_lines)
        print(row_num, col_num)



def labelme_save_file(input_path, image_name, label_path):
    img = cv2.imread(input_path)
    h, w, _ = img.shape
    print(img.shape)
    row_lines, col_lines = table_line(img[..., ::-1], size=(1024, 1024), hprob=0.5, vprob=0.5)

    print("横线{}条，竖线{}条".format(len(row_lines), len(col_lines)))

    new_row_lines = filter_lines(row_lines, col_lines)
    new_col_lines = filter_lines(col_lines, row_lines)
    print("过滤后，横线{}条，竖线{}条".format(len(row_lines), len(col_lines)))
    sorted_row_lines = sort_rowlines(new_row_lines)
    sorted_col_lines = sort_collines(new_col_lines)

    sorted_row_lines = row_line_merge(sorted_row_lines)
    # sorted_col_lines = row_line_merge(sorted_col_lines)
    sorted_col_lines = filter_col_lines(h, sorted_col_lines)
    print("合并后，横线{}条，竖线{}条".format(len(sorted_row_lines), len(sorted_col_lines)))

    labelme_json(sorted_row_lines, sorted_col_lines, h, w, img, image_name,label_path)


def filter_col_lines(h, sorted_col_lines):
    filter_index = []
    for i,line in enumerate(sorted_col_lines):
        x_max = max(int(line[0]), int(line[2]))
        x_min = min(int(line[0]), int(line[2]))
        y_max = max(int(line[1]), int(line[3]))
        y_min = min(int(line[1]), int(line[3]))
        y_min = y_max - y_min
        x_min = x_max - x_min
        # if x_min < 10:
        #     filter_index.append(i)
        # else:
        if y_max > h/2 and y_min < 180:
            filter_index.append(i)
        else:
            continue

    #filter_index = list(set(filter_index))
    filter_index.reverse()
    print("删除索引：", filter_index)
    for idx in filter_index:
        del sorted_col_lines[idx]

    return sorted_col_lines



def labelme_json(row_lines, col_lines, h, w, img, image_name,label_path):
    result = []
    for row_line in row_lines:
        class_points = {
            "label": "0",
            "points": [[row_line[0], row_line[1]], [row_line[2], row_line[3]]],
            "group_id": " ",
            "shape_type": "line",
            "flags": {}
        }
        result.append(class_points)

    for col_line in col_lines:
        class_points = {
            "label": "1",
            "points": [[col_line[0], col_line[1]], [col_line[2], col_line[3]]],
            "group_id": " ",
            "shape_type": "line",
            "flags": {}
        }
        result.append(class_points)
    prediction = {"version": "4.2.10",
                  "flags": {},
                  'shapes': result,
                  "imagePath": image_name,
                  "imageData": nparray2base64(img),
                  "imageHeight": h,
                  "imageWidth": w
                  }

    with open(label_path, "w", encoding='utf-8') as g:
        json.dump(prediction, g, indent=2, sort_keys=True, ensure_ascii=False)


# 单张测试
# if __name__ == '__main__':
    # t = time.time()
    # input_path = "/Users/yanmeima/Desktop/表格检测/ht/F2012201923500180118+保单缺失_6.png"
    # output_path = '/Users/yanmeima/Desktop/F2012201923500180118+保单缺失_6.png'
    # # input_path = "train/dataset-line/0/1.jpg"
    # # output_path = ""
    # main(input_path, output_path)

    # img_path = "data/djz/images/2022011181122108943100009K6-1.jpg"
    # label_dir = "/Users/yanmeima/Desktop/djz_labelme/test/"
    # label_path = os.path.join(label_dir + "2022011181122108943100009K6-1.json")
    # file = "2022011181122108943100009K6-1.jpg"
    # labelme_save_file(img_path, file, label_path)



if __name__=='__main__':
    input_dir = "/Users/yanmeima/Desktop/djz_labelme/lb/images/"
    output_dir = "data/djz/line/"
    label_dir = "/Users/yanmeima/Desktop/djz_labelme/lb/label/"
    files = os.listdir(input_dir)
    for file in files:
        print(file)
        t = time.time()
        if file != '.DS_Store':
            name, ext = os.path.splitext(file)
            img_path = os.path.join(input_dir + file)
            # output_path = os.path.join(output_dir + name + '-line' + '.png')
            # main(img_path, output_path)

            label_path = os.path.join(label_dir + name + ".json")
            labelme_save_file(img_path, file, label_path)
