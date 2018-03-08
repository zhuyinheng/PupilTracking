#coding=utf-8
import os
import cv2
import numpy as np
import platform
import tensorflow as tf
import random
import time
ratio=1
def walkaround(LPWPath):
    with open("./LPW/filestable.LPW","r") as f:
        lines=f.readlines()
        return [line.replace("\n","").split(" ") for line in lines]

if __name__ == '__main__':
    if platform.system()=='Windows':
        LPW=r"./LPW"
    else:
        LPW=r"./LPW"
    filemapping=walkaround(LPW)
    writer= tf.python_io.TFRecordWriter("./dataset_1_to_6")
    # cc=0;
    count=0
    for i in filemapping:
        count=count+1
        if count>6:
            break
        # if cc>0:
        #     break
        InputVideo=cv2.VideoCapture(i[0])
        GT=open(i[1],"r")
        print(i[1])
        for j in range(0,1999):
            try:
                frame=InputVideo.read()
                frame = cv2.resize(frame[1], (224, 224))
                frame=np.array(frame, dtype=np.uint8)
                line=GT.readline().replace("\n","").split(" ")
                x=int(float(line[0])/ratio);y=int(float(line[1])/ratio)
            except:
                print ("error")
                break
            t = time.clock()

            example = tf.train.Example(features=tf.train.Features(feature={
                "x": tf.train.Feature(float_list=tf.train.FloatList(value=[x])),
                "y": tf.train.Feature(float_list=tf.train.FloatList(value=[y])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[frame.tostring()]))
            }))
            writer.write(example.SerializeToString())
            # GenSubimg(frame,x,y)
            # print(time.clock() - t)



        # cc=cc+1
    writer.close()



