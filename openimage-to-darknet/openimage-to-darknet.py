#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:00:59 2019

@author: seungmin
"""

import os
import argparse
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--file_path", type=str, help="이미지 폴더 경로")
    
    parser.add_argument("-o", "--object_number", type=int, help="객체명 넘버링")
    
    opt = parser.parse_args()
    print(opt)

# 
def read_folder(path, extension):
    file_list = os.listdir(path)
    file_list_ext = [file for file in file_list if file.endswith(extension)]
    return sorted(file_list_ext)

# image width, height 출력
def get_image(image_file):
    image = Image.open(image_file + ".jpg")
    w, h = image.size
    return float(w), float(h)

def darknet_format_maker(img_w, img_h, left, top, right, bottom):
    
    left = float(left)
    top = float(top)
    right = float(right)
    bottom = float(bottom)
    
    x = (left + ((right-left)/2))/img_w
    y = (top + (bottom-top)/2)/img_h
    box_w = (right-left)/img_w
    box_h = (bottom-top)/img_h
    
    return x, y, box_w, box_h

list_of_images = read_folder(opt.file_path, ".jpg")
list_of_labels = read_folder(opt.file_path + 'Label/', ".txt")
print(list_of_labels)

try:
    os.mkdir('new_Label')
except FileExistsError:
    pass

for label in list_of_labels:
    
    # 변수 정의
    image_width, image_height = get_image(opt.file_path + label.split(".")[-2])
    print(image_width, image_height)
    
    # 이미지 데이터 경로 텍스트 파일 읽기
    with open(opt.file_path + 'Label/' + label, 'r') as file:
        lines = file.readlines()
        
        annotations = []
        for line in lines:
            object_name, left_x, top_y, right_x, bottom_y = line.split(" ")
            
            object_int = object_name.replace(object_name, str(opt.object_number))
            
            x, y, box_w, box_h = darknet_format_maker(image_width, image_height,
                                                      left_x, top_y, right_x, bottom_y)
            
            new_line = " ".join(map(str, [object_int, round(x,6), round(y,6), round(box_w,6), round(box_h,6)]))
            line = line.replace(line, new_line)
            annotations.append(line)
        print(annotations)

    # 수정된 텍스트 파일 저장
    
        with open('new_Label/'+ label, 'w') as file:
            for i, obj in enumerate(annotations):
                if i == len(annotations)-1:
                    file.write("%s" % obj)
                else:
                    file.write("%s\n" % obj)