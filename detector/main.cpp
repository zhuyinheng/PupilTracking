#include "Multithreshould.h"
#include <iostream>
#include "algo.h"
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <queue>
#include <algorithm>
#include "SK_tracker.h"
//#define DEBUG_USER
ofstream EstimationData;
using namespace std;
using namespace cv;
const int ratio = 1;
int FrameCount = 0;
double PointDIS(Point2f A, Point2f B) { return sqrt(((A.x - B.x)*(A.x - B.x) + (A.y - B.y)*(A.y - B.y))); }
bool BoundaryCheck(Rect2f roi, Size framesize)
{
	if (roi.x + roi.width >= framesize.width)
	{
		return false;
	}
	if (roi.y + roi.height >= framesize.height)
	{
		return false;
	}
	if (roi.x < 0)
	{
		return false;
	}
	if (roi.y < 0)
	{
		return false;
	}
	return true;
}
void ReadGroundTruth(string FilenameOfGroundTruth, vector<Point2f> &GroundTruth)
{
	ifstream file;
	file.open(FilenameOfGroundTruth.c_str());
	float x, y;
	while (!file.eof())
	{
		file >> x >> y;
		GroundTruth.push_back(Point2f(x, y));
	}
	return;
}
void WorkOnVideo(string FilenameOfVideo, string FilenameOfGroundTruth)
{
	VideoCapture InputVideo;

	vector<Point2f> GroundTruth;
	InputVideo.open(FilenameOfVideo);
	ReadGroundTruth(FilenameOfGroundTruth, GroundTruth);
#ifdef DEBUG_USER
	int ELSE_SUSS=0;
	int SBD_SUSS = 0;
#endif // DEBUG_USER
	Point2f lastResult(255.0f,255.0f);
	for (int i = 0; i < InputVideo.get(CV_CAP_PROP_FRAME_COUNT); i++)
	{
		
		Mat frame_raw, frame_gray;
		InputVideo >> frame_raw;
		// ACCELERATE THROUGH RESIZE IMG
		//resize(frame_raw, frame_raw, frame_raw.size() / ratio);
		cvtColor(frame_raw, frame_gray, CV_BGR2GRAY);

		Multithreshould detector(ratio);
		Point2f Result_SBD;
		Point2f Result_ELSE;
		Rect Bbox_SBD, Bbox_ELSE, Bbox_last;
		Result_ELSE = ELSE::run(frame_gray).center;
		Result_SBD = detector.optimize(frame_gray);
		EstimationData << FilenameOfVideo << " " << i << " " << Result_ELSE.x << " " << Result_ELSE.y << " " << Result_SBD.x << " " << Result_SBD.y << " " << GroundTruth[i].x << " " << GroundTruth[i].y << endl;

	}
}
void WorkOnDemo(string FilenameOfVideo)
{
	VideoCapture InputVideo;

	InputVideo.open(FilenameOfVideo);

	Point2f lastResult(255.0f,255.0f);
	for (int i = 0; i < InputVideo.get(CV_CAP_PROP_FRAME_COUNT); i++)
	{
		
		Mat frame_raw, frame_gray;
		InputVideo >> frame_raw;
		//resize(frame_raw, frame_raw, frame_raw.size() / ratio);
		cvtColor(frame_raw, frame_gray, CV_BGR2GRAY);

		Multithreshould detector(ratio);
		Point2f Result_SBD;
		Point2f Result_ELSE;
		Rect Bbox_SBD, Bbox_ELSE, Bbox_last;
		Result_ELSE = ELSE::run(frame_gray).center;
		Result_SBD = detector.optimize(frame_gray);
		EstimationData << FilenameOfVideo << " " << i << " " << Result_ELSE.x << " " << Result_ELSE.y << " " << Result_SBD.x << " " << Result_SBD.y endl;

	}
}
int main()
{

	ifstream file;
	string FilenameOfVideo, FilenameOfGroundTruth;
#ifdef _WIN32
	file.open("./videofiles.txt");
	EstimationData.open("./temp");
#else
	file.open("./videofiles.txt");
	EstimationData.open("./temp");
#endif
	while (!file.eof())
	{
		file >> FilenameOfVideo >> FilenameOfGroundTruth;
		if(file.eof())
			break;
		WorkOnVideo(FilenameOfVideo, FilenameOfGroundTruth);
#ifdef _WIN32
		break;
#endif
	}
	EstimationData.close();
	return 0;
}
