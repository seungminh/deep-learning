#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:00:59 2019

@author: seungmin
"""

import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--file_path", type=str, help="이미지 폴더 경로")
    
    parser.add_argument("-c", "--custom_path", type=str, help="커스텀 데이터 폴더 경로")
    
    parser.add_argument("-n", "--train_valid", type=str, help="train or valid")
    
    opt = parser.parse_args()
    print(opt)


# 
def read_folder(path):
    
    file_list = os.listdir(path)
    ext_list = []
    for file in file_list:
        name, ext = os.path.splitext(file)
        ext_list.append(ext)
        
    ext_list = list(set(list(filter(None, ext_list))))
    
    file_list_ext = []
    for file in file_list:
        for ext in ext_list:
            if file.endswith(ext):
                file_list_ext.append(file)
    
    return sorted(file_list_ext), sorted(ext_list)


list_of_images = read_folder(opt.file_path)[0]
print(list_of_images)


with open("./"+ opt.train_valid + ".txt", 'w') as file:
    
    for i, image in enumerate(list_of_images):
        
        if i == len(list_of_images)-1:
            file.write(opt.custom_path + image)
        else:
            file.write(opt.custom_path + image + "\n")