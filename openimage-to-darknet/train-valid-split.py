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
    
    parser.add_argument("-p", "--file_path", type=str, nargs='*', help="이미지 폴더 경로")
    
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


list_of_images = []
for one_folder in opt.file_path:
    folder_images = read_folder(one_folder)[0]
    list_of_images.append(folder_images)

print(list_of_images)



    

with open("./"+ opt.train_valid + ".txt", 'w') as file:
    
    for h, one_folder in enumerate(opt.file_path):
        
        for i, image in enumerate(list_of_images[h]):
            
            if h == opt.file_path[-1] and i == list_of_images[opt.file_path[-1]]:
                file.write(os.path.abspath(opt.file_path[h]) + "/" + image)
                
            else:
                file.write(os.path.abspath(opt.file_path[h]) + "/" + image + "\n")