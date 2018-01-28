#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <deque>
#include <queue>
#include <algorithm>
using namespace std;
using namespace cv;
class Multithreshould
{
	struct Center
	{
		Point2d location;
		double radius;
		double confidence;
	};
	double KeyPointDIS(KeyPoint A, KeyPoint B) { return sqrt(((A.pt.x - B.pt.x)*(A.pt.x - B.pt.x) + (A.pt.y - B.pt.y)*(A.pt.y - B.pt.y))); }
	Mat inverse_gray(Mat& A);
public:
	
	SimpleBlobDetector::Params blob_black,blob_white;
	SimpleBlobDetector::Params params;
	Multithreshould(int ratio);
	void findBlobs(Mat& _image, Mat& _binaryImage, std::vector<Center> &centers) const;
	void detect(Mat& image, std::vector<cv::KeyPoint>& keypoints);
	Point2f optimize(Mat& image);
	KeyPoint seletBestPupil(vector<KeyPoint> &KeyPoints);
	~Multithreshould();
};

