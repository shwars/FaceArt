
import argparse
import yaml
import os
import cognitive_face as face
import mPyPl as mp

script_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(script_dir,'config.yaml')) as f:
    conf = yaml.load(f)

parser = argparse.ArgumentParser()

parser.add_argument("dir",help="Directory of people photos",default=".")
parser.add_argument("--facegroup",help="Name of face group",default="maingroup")

args = parser.parse_args()

face.BaseUrl.set(conf['FaceApi']['Endpoint'])
face.Key.set(conf['FaceApi']['Key'])

classes = mp.get_classes(args.dir)
data = mp.get_datastream(args.dir,classes=classes) | mp.as_list

print("Person Group Trainer Utility")
print(" + found {} people".format(len(classes)))
print(" + Creating face group {}".format(args.facegroup))
face.person_group.create(args.facegroup,name=args.facegroup)

people = {}

for p in classes.keys():
    photos = data | mp.filter('class_name',lambda x:x==p) | mp.as_list
    print("Adding person {} - {} pics".format(p,len(photos)))
    pers = face.person.create(args.facegroup,p)
    people[pers['personId']]=p
    for x in photos:
        print(" + Adding photo {}".format(x['filename']),end='')
        try:
            face.person.add_face(x['filename'],args.facegroup,pers['personId'])
            print("-> ok")
        except:
            print("-> error")

print("Training...")
face.person_group.train(args.facegroup)
print("All done")

print('The following people were added to persongroup')
for k,v in people.items():
    print("{}: {}".format(k,v))

