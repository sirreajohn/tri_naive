import json
import cv2 
import numpy as np

def generate_merged_image(input_path):
    facehair_path = input_path + '/facehair.png'
    garment_top_path = input_path + '/garment_top.png'
    positions_path = input_path + '/positions.json'
    facehair_img = cv2.imread(facehair_path,cv2.IMREAD_UNCHANGED)
    garment_top_img = cv2.imread(garment_top_path,cv2.IMREAD_UNCHANGED)
    positions = json.load(open(positions_path))

    x = positions['x']
    y = positions['y']
    w = positions['w']
    h = positions['h']
    output_img = garment_top_img
    output_img[y:y+h,x:x+w] = facehair_img
    cv2.imwrite(input_path + '/merged.png',output_img)

generate_merged_image("/outputpairs/1")