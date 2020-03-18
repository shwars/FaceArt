
import argparse
import yaml
import os
import mPyPl as mp
import cv2
import json
import numpy as np
import functools
import random
import pickle

parser = argparse.ArgumentParser("Face Dataset Extractor")

parser.add_argument("input_dir",help="Input directory of pictures with JSON descriptions",default=".")
parser.add_argument("output_dir",help="Output directory",default="./out")
parser.add_argument("--size",type=int,help="Size of output images",default=600)
parser.add_argument("--template",type=str,help="Filename template to use, {} means number, extension defines file type",default="face{}.jpg")
parser.add_argument("--ignore_small",help="Ignore small faces",action='store_true')
parser.add_argument("--ignore_multiface",help="Ignore pictures with many faces present",action='store_true')
args = parser.parse_args()

dir = args.input_dir
out_dir = args.output_dir
size=args.size

target_triangle = [(x/300*size,y/300*size) for (x,y) in [(130,120),(170,120),(150,160)]]

print("Face Dataset Generator")

print(" + Loading descriptions from {}".format(dir))

def loadjs(fn):
    with open(fn) as f:
        return json.load(f)

min_size = size/3 if args.ignore_small else 0
max_faces_no = 2 if args.ignore_multiface else 99999

data = (
        mp.get_files(dir, ext='.json')
        | mp.as_field('filename')
        | mp.apply('filename', 'descr', loadjs)
        | mp.filter('descr', lambda x: len(x)>0 and len(x)<max_faces_no)
        | mp.unroll('descr')
        | mp.filter('descr', lambda x: abs(x['faceAttributes']['headPose']['yaw']) < 15 and abs(x['faceAttributes']['headPose']['pitch']) < 15)
        | mp.filter('descr',
                    lambda x: x['faceLandmarks']['pupilRight']['x'] - x['faceLandmarks']['pupilLeft']['x'] > min_size)
        | mp.as_list)

print(" + Found {} faces".format(len(data)))

print(" + Storing dataset...")

@mp.Pipe
def savepics(seq,fn):
    for i,im in enumerate(seq):
        cv2.imwrite(fn.format(i),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))

def get_transform(descr):
    f = descr['faceLandmarks']
    mc_x = (f['mouthLeft']['x']+f['mouthRight']['x'])/2.0
    mc_y = (f['mouthLeft']['y'] + f['mouthRight']['y']) / 2.0
    return cv2.getAffineTransform(np.float32([(f['pupilLeft']['x'],f['pupilLeft']['y']),(f['pupilRight']['x'],f['pupilRight']['y']),(mc_x,mc_y)]),
                                np.float32(target_triangle))

def transform(args):
    image,descr = args
    tr = get_transform(descr)
    return cv2.warpAffine(image,tr,(size,size))

(data
| mp.apply('filename','image',lambda x: cv2.cvtColor(cv2.imread(os.path.splitext(x)[0]+'.jpg'),cv2.COLOR_BGR2RGB))
| mp.apply(['image','descr'],'face',transform)
| mp.select_field('face')
| savepics(os.path.join(args.output_dir,args.template))
)

print ("All done")

