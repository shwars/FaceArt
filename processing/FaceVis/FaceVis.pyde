import json

facelst = []

def drawface(f):
    background(255)
    for x in f:
        ellipse(x[0],x[1],3,3)

def setup():
    global facelst
    size(1000,1000)
    fill(0)
    with open('z:/temp/facepts.json') as f:
        facelst = json.load(f)
    drawface(facelst[0])
    
cnt = 0
fcnt = 0
    
def draw():
    global cnt,fcnt
    cnt+=1
    if cnt%30==0:
       drawface(facelst[fcnt])
       fcnt+=1
       if fcnt==len(facelst):
           fcnt=0 
