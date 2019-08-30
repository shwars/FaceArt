
import argparse
import yaml
import os
import mPyPl as mp
from mPyPl.utils.image import show_images, im_resize
import cv2
import json
import numpy as np
import functools
import random

parser = argparse.ArgumentParser()

parser.add_argument("dir",help="Directory of pictures to be processed",default=".")
parser.add_argument("--person",help="Person ID",default="")
parser.add_argument("--large-fit",help="Use large fit, i.e. make face smaller")
parser.add_argument("--num",type=int,help="Number of pictures to generate",default=1)

args = parser.parse_args()

dir = 'z:/temp/pics'
person_id = '764f5c61-1d79-4ee4-870c-4d6190b0c22a'
person_id = 'bc708829-b13e-470b-b64d-749dd81e03ae'
person_id = 'ac4861f7-ff89-4f68-909b-99d49bd1ea9e' # Tanya
person_id = '6af1e28f-b74b-4856-8e73-d4f01f6cb560' # Aki
person_id = ''
dir = args.dir
person_id = args.person

print("Opaque Portrait Generator")

def loadjs(fn):
    with open(fn) as f:
        return json.load(f)

def cutout(args):
    im,js = args
    expand = 0.3
    # print(im.shape)
    im_h,im_w = im.shape[0:2]
    x,y,w,h = [ js['faceRectangle'][t] for t in ['top','left','height','width']]
    x_expand = int(expand*w)
    y_expand = int(expand*h)
    # cv2.rectangle(im,(y,x),(y+h,x+w),(255,255,255),3)
    return im[max(0,x-4*x_expand):min(im_h,x+w+2*x_expand),max(0,y-y_expand):min(im_w,y+h+y_expand)]

def merge(images,wts=None):
    res = np.zeros_like(images[0],dtype=np.float32)
    if wts is None:
        wts = np.ones(len(images))
    wts /= np.sum(wts)
    for n,i in enumerate(images):
        res += wts[n]*i.astype(np.float32)
    return res.astype(np.uint)

data = (
    mp.get_files(dir,ext='.json')
    | mp.as_field('filename')
    | mp.apply('filename','descr',loadjs)
    | mp.unroll('descr')
    | mp.filter('descr',lambda x: person_id=="" or ('candidates' in x and person_id in [z['personId'] for z in x['candidates']]))
    | mp.filter('descr',lambda x: abs(x['faceAttributes']['headPose']['yaw'])<15 and abs(x['faceAttributes']['headPose']['pitch'])<15)
    | mp.as_list)

print("Found {} faces".format(len(data)))

def get_transform(descr):
    f = descr['faceLandmarks']
    mc_x = (f['mouthLeft']['x']+f['mouthRight']['x'])/2.0
    mc_y = (f['mouthLeft']['y'] + f['mouthRight']['y']) / 2.0
    return cv2.getAffineTransform(np.float32([(f['pupilLeft']['x'],f['pupilLeft']['y']),(f['pupilRight']['x'],f['pupilRight']['y']),(mc_x,mc_y)]),
                                np.float32([(80,150),(120,150),(100,190)]))

def transform(args):
    image,descr = args
    tr = get_transform(descr)
    return cv2.warpAffine(image,tr,(200,300))

(data
| mp.pshuffle
| mp.take(15)
| mp.apply('filename','image',lambda x: cv2.cvtColor(cv2.imread(os.path.splitext(x)[0]+'.jpg'),cv2.COLOR_BGR2RGB))
| mp.apply(['image','descr'],'face',transform)
| mp.apply('face','facesmall',functools.partial(im_resize,size=(100,150)))
| mp.apply('descr','ypr',lambda x: "Y={},P={},R={}".format(x['faceAttributes']['headPose']['yaw'],x['faceAttributes']['headPose']['pitch'],x['faceAttributes']['headPose']['roll']))
| mp.select_field('facesmall')
| mp.pexec(functools.partial(show_images,cols=3)))

imgs = (data
| mp.pshuffle
| mp.take(30)
| mp.apply('filename','image',lambda x: cv2.cvtColor(cv2.imread(os.path.splitext(x)[0]+'.jpg'),cv2.COLOR_BGR2RGB))
| mp.apply(['image','descr'],'face',transform)
| mp.apply('face','facesmall',functools.partial(im_resize,size=(100,150)))
| mp.select_field('facesmall')
| mp.as_list)

(range(10)
 | mp.select(lambda _: merge(imgs,np.random.random(len(imgs))))
 | mp.pexec(functools.partial(show_images,cols=2)))


def generate_img(data):
    n = random.randint(3,30)
    x = (data
         | mp.pshuffle
         | mp.take(n)
         | mp.apply('filename','image',lambda x: cv2.cvtColor(cv2.imread(os.path.splitext(x)[0]+'.jpg'),cv2.COLOR_BGR2RGB))
         | mp.apply(['image','descr'],'face',transform)
         | mp.apply('face','facesmall',functools.partial(im_resize,size=(100,150)))
         | mp.select_field('facesmall')
         | mp.as_list)
    return merge(x,np.random.random(len(x)))

(range(10)
 | mp.select(lambda _: generate_img(data))
 | mp.pexec(functools.partial(show_images,cols=2)))

## Generate transformed face points

def calc_pts(args):
    descr,transform = args
    pts = descr['faceLandmarks']
    ar = np.array([[p['x'],p['y'],1.] for p in pts.values()])
    m = transform@ar.T
    return list(map(list,zip(*m)))

@mp.Pipe
def writeout(s,fn):
    with open(fn,'w') as f:
        f.write('[\n')
        f.write(",\n".join(map(str,s)))
        f.write('\n]\n')

(data
    | mp.apply('descr','transform', get_transform)
    | mp.apply(['descr','transform'],'points',calc_pts)
    | mp.select_field('points')
    | writeout('z:/temp/facepts.json'))
