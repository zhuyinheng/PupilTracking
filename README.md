# PLEASE CHECK THE GITHUB FOR THE LATEST VERSION:
[Paper](https://arxiv.org/abs/1805.00311)
# REQUIREMENT:
- opencv 3.3.0
- tensorflow 1.2.1
- tensorflow slim
- python 3.6

# HOW TO USE IT
1. set up opencv dir in ./detector/CMakeLists.txt: 
```
SET("OpenCV_DIR" "<path to the /opencv/build/>")
```

2. download the [LPW](http://datasets.d2.mpi-inf.mpg.de/tonsen/LPW.zip) dataset and decompress into ./LPW

3. generate videofile.txt
```
python gen_videofiles.py
```

4. compile detector:
```
cd detector
make
```
5. run the detector
```
./detector/PupilDetection
```

6. wait until PupilDetection finish

7. download [pretrained model](https://drive.google.com/file/d/1f6AcGv_7w6o5wr24cIId9wN56wedv5YY/view?usp=sharing) and decompress into ./pretrain/

8. run evaluator
```
python evaluator/evaluator.py
```

9. check the result in ./result

# Explaination of the result
## Structure:

```
- result
    - <alpha><Is_average_filter><videonumber> (it's the result of each video, so there are 64 files like this. e.g.  0.005False60)
    - <alpha><Is_average_filter><finish time stamp>: e.g. 0.005False60time.struct_time(tm_year=2017, tm_mon=8, tm_mday=22, tm_hour=16, tm_min=12, tm_sec=26, tm_wday=1, tm_yday=234, tm_isdst=1) (it's the result of all each video)
```


## For the result of each video:

```
line 1: (e.g. ./LPW/23/2.avi) is the file path of the video
line 2: (e.g. NEEDTOIMPROVE11) represent the NumberOfFrame(upperbound)-NumberOfFrame(evaluator). It's not relevant to the paper.
line 3 - line 503: (e.g. Pixcel  242: 0.999) means the accuracy(0.999) with condition that distance(point(predict),point(groundtruth))
```

## For the result of all videos:

```
line 1: (e.g. NEEDTOIMPROVE11) represent the NumberOfFrame(upperbound)-NumberOfFrame(evaluator). It's not relevant to the paper.
line 2 - line 502: (e.g. Pixcel  242: 0.999) means the accuracy(0.999) with condition that distance(point(predict),point(groundtruth))
```

# Optional: fine-tune

1. make the dataset
```
python ./train_evaluator/makedataset.py
```
2. train
```
download pretrain vgg model and put it into /train_evaluator/pretrain_vgg
python ./train_evaluator/train.py
```

# Code reference

- `./detector/algo.h ./detector/blob_gen.h ./detector/canny_impl.h ./detector/filter_edges.h ./detector/find_best_edge.h`:We use the first part of [ElSe algorithm](https://arxiv.org/pdf/1511.06575.pdf)[1], which is based on morphologic feature as one of the answer candidates.
- `./evaluator/vgg.py`We use [VGG-16](http://arxiv.org/pdf/1409.1556.pdf)[2] architecture.
- `./LPW/` We use part of (about 1/80) [LPW dataset] to fine-tune the network, and estimate the method on this dataset.

*[1]:Fuhl, Wolfgang, et al. "Else: Ellipse selection for robust pupil detection in real-world environments." Proceedings of the Ninth Biennial ACM Symposium on Eye Tracking Research & Applications. ACM, 2016.*

*[2]:Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).*
