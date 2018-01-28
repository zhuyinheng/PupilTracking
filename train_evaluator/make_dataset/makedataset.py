#coding=utf-8
import os
import cv2
import numpy as np
import platform
import tensorflow as tf
import random
import time
from matplotlib import pyplot as plt
ratio=1
WinRadio=60
stride=int(5)
rate=1
def Cut(frame,x,y):
    return frame[y - WinRadio: y +WinRadio, x - WinRadio:x + WinRadio, 0]
def Ramdomselect(frame,x,y,minr,maxr):
    for i in range(500):
        minr=int(minr)
        maxr=int(maxr)
        random.random()
        dx=random.randint(minr,maxr)*random.choice([-1,1])
        dy=random.randint(minr,maxr)*random.choice([-1,1])
        if not (x+dx < 0 or x+dx >= 480 or y+dy < 0 or y+dy >= 640):
            return x+dx,y+dy
    return False
def Padding(frame, centerpoint):
    centerpoint = (int(centerpoint[0]), int(centerpoint[1]))
    if not (centerpoint[0] - WinRadio < 0 or centerpoint[0] + WinRadio >= 480 or centerpoint[1] - WinRadio < 0 or centerpoint[
        1] + WinRadio >= 640):
        sub = frame[centerpoint[0] - WinRadio:centerpoint[0] + WinRadio, centerpoint[1] - WinRadio: centerpoint[1] + WinRadio, :]
    else:
        radiu = WinRadio - max(0 - (centerpoint[0] - WinRadio), 0 - (centerpoint[1] - WinRadio), centerpoint[0] + WinRadio - 480,
                         centerpoint[1] + 60 - 480)
        t = frame[centerpoint[0] - radiu:centerpoint[0] + radiu, centerpoint[1] - radiu: centerpoint[1] + radiu, :]
        v=max(frame.flatten())
        sub = np.lib.pad(t, ((WinRadio - radiu, WinRadio - radiu), (WinRadio - radiu, WinRadio - radiu), (0, 0)), 'constant',
                         constant_values=v)
    return sub
def GenSubimg(frame,x,y):
    frame = np.array(frame[1], dtype=np.uint8)
    c=[(0,5),(5,60),(60,640)]
    for i in range(3):
        center=Ramdomselect(frame,x, y, c[i][0],c[i][1])
        if center is False:
            continue
        img=Padding(frame, center)

        raw=img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw]))
        }))
        writer.write(example.SerializeToString())

def walkaround(LPWPath):
    with open("./videofiles.txt","r") as f:
        lines=f.readlines()
        return [line.replace("\n","").split(" ") for line in lines]
if __name__ == '__main__':
    if platform.system()=='Windows':
        LPW=r"./LPW"
    else:
        LPW=r"./LPW"
    filemapping=walkaround(LPW)
    writer= tf.python_io.TFRecordWriter("./dataset.tfrecord")
    # cc=0;
    for i in filemapping:
        # if cc>0:
        #     break
        InputVideo=cv2.VideoCapture(i[0])
        GT=open(i[1],"r")
        for j in range(0,2001):
            try:
                frame=InputVideo.read()
                line=GT.readline().replace("\n","").split(" ")
                if random.random()>rate:
                    continue
                x=int(float(line[0])/ratio);y=int(float(line[1])/ratio)
            except:
                print ("error")
                break
            t = time.clock()
            GenSubimg(frame,x,y)
            print(time.clock()-t)
        # cc=cc+1
    writer.close()



