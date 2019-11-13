#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:48:52 2019

@author: seungmin
"""

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="텍스트 파일 위치 & 파일명")
    parser.add_argument("--replace_A", type=str, help="바꿔야 할 텍스트")
    parser.add_argument("--replace_B", type=str, help="올바른 텍스트")
    opt = parser.parse_args()
    print(opt)

# 이미지 데이터 경로 텍스트 파일 읽기
with open(opt.file_path, 'r') as file :
  filedata = file.read()

# 텍스트 파일 내부 수정
filedata = filedata.replace(opt.replace_A, opt.replace_B)

# 수정된 텍스트 파일 저장
with open('./name.txt', 'w') as file:
  file.write(filedata)

