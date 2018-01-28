#coding=utf-8
import platform
import os
import cv2
from scipy import interpolate
from skimage import transform
import math
import numpy as np
import time
import tensorflow as tf
import vgg

ErrorN_G=np.zeros((501),dtype=np.uint32)
needto_G=0
videocount=0
tframecount=0
LAST_LOSS=0.005
IS_AVERAGE=False
def caldis(A,B):
    return math.sqrt((A[0]-B[0])*(A[0]-B[0])+(A[1]-B[1])*(A[1]-B[1]))
def calcore(img):
    score=sess.run(prob,feed_dict={images:np.reshape(img,[1,224,224,3])})
    return score[0][0]
def GetSubIMG(frame,centerpoint):
    centerpoint=(int(centerpoint[0]),int(centerpoint[1]))
    if not(centerpoint[0] - 60 < 0 or centerpoint[0] + 60 >= 480 or centerpoint[1] - 60 < 0 or centerpoint[1] + 60 >= 640):
        sub=frame[centerpoint[0] - 60:centerpoint[0] + 60,centerpoint[1] - 60 : centerpoint[1] + 60,:]
    else:
        radiu=60-max(0-(centerpoint[0] - 60),0-(centerpoint[1] - 60),centerpoint[0] + 60-480,centerpoint[1] + 60-480)
        t=frame[centerpoint[0] - radiu:centerpoint[0] + radiu,centerpoint[1] - radiu : centerpoint[1] + radiu,:]
        sub=np.lib.pad(t,((60-radiu,60-radiu),(60-radiu,60-radiu),(0,0)),'constant',constant_values=255)
    sub_up = transform.resize(sub, (224, 224, 3))
    return sub_up
def WorkOnSingleVideo(VideoName):
    InputVideo=cv2.VideoCapture()
    InputVideo.open(VideoName)
    ErrorN = np.zeros((501),dtype=np.uint32)
    ans = {"ELSE": (0, 0), "SBD": (0, 0), "LAST": (240,320)}
    global tframecount
    tframecount+=int(InputVideo.get(7))# 7 stand for the marco definition CV_CAP_PROP_FRAME_COUNT
    needto_L=0
    for i in range(int(InputVideo.get(7))):

        score = {"ELSE": -3, "SBD": -3, "LAST": -3}
        _,frame=InputVideo.read()
        prevAns=prevAnss.readline().split(" ")
        assert (prevAns!="")
        assert (prevAns[0] == VideoName and prevAns[1] == str(i))
        # debugging
        print(prevAns[0],VideoName,prevAns[1],i)

        # Read Ans And calculate score
        ans["ELSE"]=(float(prevAns[3]),float(prevAns[2]))
        ans["SBD"] = (float(prevAns[5]),float(prevAns[4]))
        groundtruth=(float(prevAns[7]),float(prevAns[6]))
        if ans["ELSE"] ==(-1,-1) or ans["ELSE"]==(0,0):
            score["ELSE"] = -2
        if ans["SBD"] ==(-1,-1) or ans["SBD"]==(0,0):
            score["SBD"] = -2

        score["LAST"]=calcore(GetSubIMG(frame, ans["LAST"]))*LAST_LOSS

        if score["ELSE"]!=-2:
            score["ELSE"]=calcore(GetSubIMG(frame, ans["ELSE"]))
        if score["SBD"]!=-2:
            score["SBD"]=calcore(GetSubIMG(frame, ans["SBD"]))

# select final ans
        if IS_AVERAGE is True:
            candidate=[]
            ava=0
            for k,v in score.items():
                if v>0.999:
                    candidate.append(ans[k])
                    ava+=1

            if ava>0:
                final_result=(0,0)
                for v in candidate:
                    final_result=(final_result[0]+v[0],final_result[1]+v[1])
                final_result=(final_result[0]/ava,final_result[1]/ava)
                chose="WITHINERROR"
            else:
                if score["ELSE"] >= score["SBD"] and score["ELSE"] >= score["LAST"]:
                    final_result = ans["ELSE"]
                    chose = "ELSE"
                if score["SBD"] > score["ELSE"] and score["SBD"] >= score["LAST"]:
                    final_result = ans["SBD"]
                    chose = "SBD"
                if score["LAST"] > score["ELSE"] and score["LAST"] > score["SBD"]:
                    final_result = ans["LAST"]
                    chose = "LAST"
        else:
            if score["ELSE"] >= score["SBD"] and score["ELSE"] >= score["LAST"]:
                final_result = ans["ELSE"]
                chose = "ELSE"
            if score["SBD"] > score["ELSE"] and score["SBD"] >= score["LAST"]:
                final_result = ans["SBD"]
                chose = "SBD"
            if score["LAST"] > score["ELSE"] and score["LAST"] > score["SBD"]:
                final_result = ans["LAST"]
                chose = "LAST"

# DEBUGGING
        dis1=caldis(ans["ELSE"],groundtruth)
        dis2 = caldis(ans["SBD"], groundtruth)
        dislast=caldis(ans["LAST"],groundtruth)
        if min(dislast,dis1,dis2)<=5 and caldis(final_result,groundtruth)>5:
            for k,v in ans.items():
                print(k,v)
            print("Result:",final_result,"GT",groundtruth)
            global needto_G
            needto_G+=1
            needto_L+=1
#
        ans["LAST"]=final_result
        er=caldis(final_result,groundtruth)
        if er<=500:
            ErrorN[int(er)] += 1
            ErrorN_G[int(er)] += 1
    for i in range(500):
        ErrorN[i+1]+=ErrorN[i]
# PRINT THE RESULT OF EACH VIDEO
    if platform.system()=="Windows":
        print("Windows is not supported\n")
        exit()
        # result_local=os.path.join(LPW,str(LAST_LOSS)+str(IS_AVERAGE)+str(videocount)+".txt")
    else:
        result_local=os.path.join(os.path.join(RES,str(LAST_LOSS)+str(IS_AVERAGE)+str(videocount)))
    with open(result_local,"w") as f:
        f.write(VideoName+'\n')
        f.write("NEEDTOIMPROVE"+str(needto_L)+"\n")
        for i in range(500):
            f.write("Pixcel  "+str(i)+": "+str(ErrorN[i]/int(InputVideo.get(7)))+"\n")

if __name__ == '__main__':
# DEAL WITH THE FILEPATH
    if platform.system()=="Windows":
        print("Windows is not supported!")
    else:
        RES=r"./Result"
        f=open("./videofiles","r")
        prevAnss = open("./temp", "r")
        checkpoint_path = r"./pretrain/model.ckpt-18979"

# DEFINE THE NET
    slim = tf.contrib.slim
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    predictions, end_points = vgg.vgg_16(images, num_classes=2, is_training=False)
    prob=tf.nn.softmax(predictions)
    variables_to_restore = slim.get_model_variables()
    init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint_path, variables_to_restore)

# LOOP FOR EACH VIDEO
    with tf.Session() as sess:
        sess.run(init_assign_op, init_feed_dict)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:

            lines=f.readlines()
            for line in lines:
                line=line.split(" ")
                WorkOnSingleVideo(line[0])
                videocount+=1
    

# PRINT THE RESULT OF EACH VIDEO
            if platform.system()=="Windows":
                # result_local=os.path.join(LPW,str(LAST_LOSS)+str(IS_AVERAGE)+str(time.localtime(time.time())))
                print("Windows is not supported\n")
                exit()
            else:
                result_local=os.path.join(os.path.join(RES,str(LAST_LOSS)+str(IS_AVERAGE)+str(time.localtime(time.time()))))
            with open(result_local,"w") as f:
                f.write("NEEDTOIMPROVE" + str(needto_G)+ "\n")
                f.write("Pixcel " + str(0) + ": " + str(ErrorN_G[0]/tframecount) + "\n")
                for i in range(499):
                    ErrorN_G[i+1]+=ErrorN_G[i]
                    f.write("Pixcel " + str(i+1) + ": " + str(ErrorN_G[i+1]/tframecount) + "\n")

        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)