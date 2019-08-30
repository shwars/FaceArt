
import argparse
import yaml
import os
import cognitive_face as face
import mPyPl as mp
import json

script_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(script_dir,'config.yaml')) as f:
    conf = yaml.load(f)

parser = argparse.ArgumentParser()

parser.add_argument("dir",help="Directory of pictures to be processed",default=".")
parser.add_argument("--facegroup",help="Name of face group to use for face recognition",default="")

args = parser.parse_args()

face.BaseUrl.set(conf['FaceApi']['Endpoint'])
face.Key.set(conf['FaceApi']['Key'])

print("Face API Calling Utility")

files = mp.get_files(args.dir) | mp.where(lambda x: not x.endswith('.json')) | mp.as_list

for x in files:
    print(" + {}".format(x),end='')
    jsfn = os.path.splitext(x)[0]+'.json'
    if os.path.isfile(jsfn):
        print(" -> skip")
        continue
    res = []
    try:
        res = face.face.detect(x,True,True,'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur')
    except:
        pass
    if len(res)>0 and args.facegroup!="":
        ids = [x['faceId'] for x in res]
        ids = ids[:10] # trim at 10 faces max per FaceAPI limitation
        r = face.face.identify(ids,args.facegroup)
        for x in res:
            c = [z['candidates'] for z in r if z['faceId']==x['faceId']]
            if len(c)>0: x['candidates'] = c[0]
    with open(jsfn,"w") as f:
        json.dump(res,f)
    print(" -> {} faces".format(len(res)))


