import argparse
import dlib
import os
import numpy as np
import PIL.Image
from tqdm import tqdm
import shutil

parser = argparse.ArgumentParser()

parser.add_argument("input_dir")
parser.add_argument("output_dir")
parser.add_argument("--percent",help="Percentage of width the face should occupy in the picture", default=0.2,type=float)
parser.add_argument("--crop-sqr",help="Crop images to square size",default=False,const=True,action='store_const')
parser.add_argument("--facecut",help="Cut out face",default=False,const=True,action='store_const')
parser.add_argument("--min-width",help="Minimum width of face to consider for cutting out",type=int,default=512)
parser.add_argument("--face-expand",help="Expand face rectangle by specified % of hidth/height",type=float,default=0.2)
parser.add_argument("--resize",help="Resize photos to target size",type=int)
args = parser.parse_args()

detector = dlib.get_frontal_face_detector()

def get_bounds(x,size,w):
    if size==w:
        return 0,size
    x1 = max(0,x-size//2)
    x2 = x1+size
    return x1,x2

def crop(img,rct):
    h,w = img.shape[0:2]
    size = min(h,w)
    x,y = (rct.left()+rct.right())//2,(rct.top()+rct.bottom())//2
    x1,x2 = get_bounds(x,size,w)
    y1,y2 = get_bounds(y,size,h)
    return img[y1:y2,x1:x2,:]

def facecut(img,rct,pct):
    h,w = img.shape[0:2]
    fw, fh = rct.right()-rct.left(),rct.bottom()-rct.top()
    x,y = (rct.left()+rct.right())//2,(rct.top()+rct.bottom())//2
    size = min(w,h,int(max(fw,fh)*(1+pct)))
    size2 = size//2
    if x-size2<0: x+=(size2-x)
    if x+size2>w: x-=x+size2-w
    if y-size2<0: y+=(size2-y)
    if y+size2>h: y=-y+size2-h
    return img[y-size2:y+size2,x-size2:x+size2]

def save(img,fn):
    if args.resize:
        PIL.Image.fromarray(res).resize((args.resize,args.resize)).save(fn)
    else:
        PIL.Image.fromarray(res).save(fn)


for f in tqdm(os.listdir(args.input_dir)):
    img = np.array(PIL.Image.open(os.path.join(args.input_dir,f)))
    res = detector(img,0)
    if len(res)==1:
        w = res[0].right()-res[0].left()
        p = w/img.shape[1]
        if (p>args.percent):
            if args.crop_sqr:
                res = crop(img,res[0])
                save(res,os.path.join(args.output_dir,f))
            elif args.facecut:
                res = facecut(img,res[0],args.face_expand)
                if res.shape[0]>=args.min_width:
                    save(res,os.path.join(args.output_dir,f))
            else:
                shutil.copy(os.path.join(args.input_dir,f),
                        os.path.join(args.output_dir,f))
