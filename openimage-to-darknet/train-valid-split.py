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
def read_folder(path, extension):
    file_list = os.listdir(path)
    file_list_ext = [file for file in file_list if file.endswith(extension)]
    return sorted(file_list_ext)


list_of_images = read_folder(opt.file_path, ".jpg")
print(list_of_images)

with open("./"+ opt.train_valid + ".txt", 'w') as file:
    
    for i, image in enumerate(list_of_images):
        
        if i == len(list_of_images)-1:
            file.write(opt.custom_path + image)
        else:
            file.write(opt.custom_path + image + "\n")