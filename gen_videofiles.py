#coding=utf-8
import os
import platform
import sys

rootdir= './LPW'

f=open("videofiles.txt","w")
for i in range(1,24):
    j=os.listdir(os.path.join(rootdir,str(i)))
    for k in j:
        if os.path.splitext(k)[1] == '.avi':
            f.write(os.path.join(rootdir,str(i),k)+' '+os.path.join(rootdir,str(i),os.path.splitext(k)[0]+'.txt')+'\n')