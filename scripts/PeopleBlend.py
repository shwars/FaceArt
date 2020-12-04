
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
import cognitive_face as face
import pickle

script_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(script_dir,'config.yaml')) as f:
    conf = yaml.load(f)

parser = argparse.ArgumentParser()

parser.add_argument("input_dir",help="Directory of pictures to be processed",default=".")
parser.add_argument("output_dir",help="Directory of pictures where to put result",default=".")
parser.add_argument("--name-template",help="Filename template to use, {} is substituted for number",default="out{}.jpg")
#parser.add_argument("--large-fit",help="Use large fit, i.e. make face smaller")
parser.add_argument("--size",type=int,help="Size of output images",default=600)
parser.add_argument("--num",type=int,help="Number of pictures to generate",default=1)
parser.add_argument("--mix",type=int,help="Number of pictures to mix (defaults to all)",default=999)
parser.add_argument("--nocache",help="Do not use face api caching",action='store_true')
parser.add_argument("--newcache",help="Rewrite the cache file",action='store_true')
parser.add_argument("--nosign",help="Do not sign photos",action='store_true')
parser.add_argument("--local-detect",help="Enable local detection of facial landmarks",action='store_true')
parser.add_argument("--signsize",help="Signature size in pixels",default=None,type=int)
args = parser.parse_args()

dir = args.input_dir
out_dir = args.output_dir
size=args.size

cache = {}
if not args.nocache and not args.newcache:
    try:
        with open(os.path.join(script_dir,'cache.pkl'),'rb') as f:
            cache = pickle.load(f)
    except:
        pass

if args.local_detect:
    import dlib
    print(" + Loading model for local detection")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

sign = None
if not args.nosign:
    sign = cv2.imread(os.path.join(script_dir,'mitya_sig.png'),cv2.IMREAD_UNCHANGED)
    sz = args.signsize or int(args.size*0.1)
    sign = cv2.resize(sign,(sz,sz))
    

face.BaseUrl.set(conf['FaceApi']['Endpoint'])
face.Key.set(conf['FaceApi']['Key'])

target_triangle = [(x/300*size,y/300*size) for (x,y) in [(130,120),(170,120),(150,160)]]

print("PeopleBlending Generator")

def merge(images,wts=None):
    res = np.zeros_like(images[0],dtype=np.float32)
    if wts is None:
        wts = np.ones(len(images))
    wts /= np.sum(wts)
    for n,i in enumerate(images):
        res += wts[n]*i.astype(np.float32)
    return res.astype(np.int32)

def detect(argms):
    fn,im = argms
    if args.local_detect:
        faces = detector(im,1)
        if len(faces)==0: return []
        lmarks = predictor(im,faces[0])
        return {
            "pupilLeft" : { "x" : (lmarks.part(37).x+lmarks.part(40).x)/2, "y" : (lmarks.part(37).y+lmarks.part(40).y)/2},
            "pupilRight" : { "x" : (lmarks.part(43).x+lmarks.part(46).x)/2, "y" : (lmarks.part(43).y+lmarks.part(46).y)/2},
            "mouthLeft" : { "x" : lmarks.part(49).x, "y" : lmarks.part(49).y },
            "mouthRight" : { "x" : lmarks.part(55).x, "y" : lmarks.part(55).y }
        }
    h = hash(fn)
    if not args.nocache and h in cache:
        return cache[h][0]['faceLandmarks']
    res = []
    try:
        res = face.face.detect(fn,True,True,'')
        if not args.nocache:
            cache[h] = res
        res = res[0]['faceLandmarks']
        #print(res[0]['faceLandmarks'])
    except:
        pass
    return res

print(" + Loading images from {}".format(dir))

data = (
    mp.get_files(dir)
    | mp.as_field('filename')
    | mp.apply_nx('filename','image',lambda x: cv2.imread(x),print_exceptions=False)
    | mp.filter('image',lambda x: x is not None)
    | mp.as_list)

print(" + Found {} images".format(len(data)))

print(" + Extracting facial landmarks...")

data = (
    data 
    | mp.apply(['filename','image'],'landmarks',detect)
    | mp.filter('landmarks',lambda x: x!=[])
    | mp.as_list
)

if not args.nocache:
    print(" + Saving cache...")
    with open(os.path.join(script_dir,'cache.pkl'),'wb') as f:
        pickle.dump(cache,f)

def transform(args):
    image,f = args
    mc_x = (f['mouthLeft']['x']+f['mouthRight']['x'])/2.0
    mc_y = (f['mouthLeft']['y'] + f['mouthRight']['y']) / 2.0
    tr = cv2.getAffineTransform(np.float32([(f['pupilLeft']['x'],f['pupilLeft']['y']),(f['pupilRight']['x'],f['pupilRight']['y']),(mc_x,mc_y)]),
                                np.float32(target_triangle))
    return cv2.warpAffine(image,tr,(size,size))

@mp.Pipe
def savepics(seq,fn):
    for i,im in enumerate(seq):
        cv2.imwrite(fn.format(i),im)

def generate_img(data):
    x = (data
         | mp.pshuffle
         | mp.take(args.mix)
         | mp.apply(['image','landmarks'],'face',transform)
         | mp.select_field('face')
         | mp.as_list)
    return merge(x,np.random.random(len(x)))

def imprint(img):
    if args.nosign:
        return img
    overlay_image = sign[..., :3]
    mask = sign[..., 3:] / 255.0
    h,w = sign.shape[0],sign.shape[1]
    x,y=args.size-h,args.size-w
    img[y:y+h, x:x+w] = (1.0 - mask) * img[y:y+h, x:x+w] + mask * overlay_image
    return img

print(" + Generating images...")

os.makedirs(out_dir,exist_ok=True)

(range(args.num)
 | mp.select(lambda _: generate_img(data))
 | mp.select(imprint)
 | savepics(os.path.join(out_dir,args.name_template)))

print ("All done")

