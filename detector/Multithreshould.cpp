#include "Multithreshould.h"
#include <iostream>
#include <iterator>
#include <limits>
//#define DEBUG_USER
//#define DEBUG_BLOB_DETECTOR
using namespace cv;
using namespace std;

Mat Multithreshould::inverse_gray(Mat & A)
{
	
	Mat img(A.size(), A.type(), Scalar(255));
	return img-A;
}

Multithreshould::Multithreshould(int ratio)
{
	
	//--------------------
	blob_black.thresholdStep = 10;
	blob_black.minThreshold = 0;
	blob_black.maxThreshold = 100;
	blob_black.minRepeatability = 2;
	blob_black.minDistBetweenBlobs = 10;
	blob_black.filterByColor = true;
	blob_black.blobColor = 0;
	blob_black.filterByArea = true;
	//------------size of absolut Area depends on the size of src-----------
	blob_black.minArea = 640 * 480 / ratio / ratio / 400;
	blob_black.maxArea = 640 * 480 / ratio/ ratio /4;

	//------------constrains that aren't used-------------------------------
	blob_black.filterByCircularity = true;
	blob_black.minCircularity = 0.3f;
	blob_black.maxCircularity = (float)1e37;
	blob_black.filterByInertia = true;
	blob_black.minInertiaRatio = 0.1f;
	blob_black.maxInertiaRatio = (float)1e37;
	blob_black.filterByConvexity = true;
	blob_black.minConvexity = 0.5f;
	blob_black.maxConvexity = (float)1e37;
	//---------------------------------------------------------------------------------
	blob_white.thresholdStep = 10;
	blob_white.minThreshold = 0;
	blob_white.maxThreshold = 100;
	blob_white.minRepeatability = 2;
	blob_white.minDistBetweenBlobs = 10;
	blob_white.filterByColor = true;
	blob_white.blobColor = 0;
	blob_white.filterByArea = true;
	//------------size of absolut Area depends on the size of src-----------
	blob_white.minArea = 5 * 5/ratio/ratio;
	blob_white.maxArea = 50 * 50 / ratio / ratio;
	//----------------------------------------------------------------------
	//------------constrains that aren't used-------------------------------
	//blob_white.filterByCircularity = false;
	//blob_white.minCircularity = 0.9f;
	//blob_white.maxCircularity = (float)1e37;
	//blob_white.filterByInertia = false;
	//blob_white.minInertiaRatio = 0.1f;
	//blob_white.maxInertiaRatio = (float)1e37;
	blob_white.filterByConvexity = false;
	blob_white.minConvexity = 0.5f;
	blob_white.maxConvexity = (float)1e37;
	//------------------------------------------------------------------------------------
	params = blob_black;
}

void Multithreshould::findBlobs(Mat& _image, Mat& _binaryImage, std::vector<Center>& centers) const
{
	//CV_INSTRUMENT_REGION()

	Mat image = _image, binaryImage = _binaryImage;
	centers.clear();

	std::vector < std::vector<Point> > contours;
	Mat tmpBinaryImage = binaryImage.clone();
	findContours(tmpBinaryImage, contours, RETR_LIST, CHAIN_APPROX_NONE);

#ifdef DEBUG_BLOB_DETECTOR
	  Mat keypointsImage;
	  cvtColor( binaryImage, keypointsImage, CV_GRAY2RGB );
	
	  Mat contoursImage;
	  cvtColor( binaryImage, contoursImage, CV_GRAY2RGB );
	  drawContours( contoursImage, contours, -1, Scalar(0,255,0) );
	  imshow("contours", contoursImage );
#endif

	for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
	{
		Center center;
		center.confidence = 1;
		Moments moms = moments(Mat(contours[contourIdx]));
		if (params.filterByArea)
		{
			double area = moms.m00;
			if (area < params.minArea || area >= params.maxArea)
				continue;
		}

		if (params.filterByCircularity)
		{
			double area = moms.m00;
			double perimeter = arcLength(Mat(contours[contourIdx]), true);
			double ratio = 4 * CV_PI * area / (perimeter * perimeter);
			if (ratio < params.minCircularity || ratio >= params.maxCircularity)
				continue;
		}

		if (params.filterByInertia)
		{
			double denominator = std::sqrt(std::pow(2 * moms.mu11, 2) + std::pow(moms.mu20 - moms.mu02, 2));
			const double eps = 1e-2;
			double ratio;
			if (denominator > eps)
			{
				double cosmin = (moms.mu20 - moms.mu02) / denominator;
				double sinmin = 2 * moms.mu11 / denominator;
				double cosmax = -cosmin;
				double sinmax = -sinmin;

				double imin = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmin - moms.mu11 * sinmin;
				double imax = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmax - moms.mu11 * sinmax;
				ratio = imin / imax;
			}
			else
			{
				ratio = 1;
			}

			if (ratio < params.minInertiaRatio || ratio >= params.maxInertiaRatio)
				continue;

			center.confidence = ratio * ratio;
		}

		if (params.filterByConvexity)
		{
			std::vector < Point > hull;
			convexHull(Mat(contours[contourIdx]), hull);
			double area = contourArea(Mat(contours[contourIdx]));
			double hullArea = contourArea(Mat(hull));
			double ratio = area / hullArea;
			if (ratio < params.minConvexity || ratio >= params.maxConvexity)
				continue;
		}

		if (moms.m00 == 0.0)
			continue;
		center.location = Point2d(moms.m10 / moms.m00, moms.m01 / moms.m00);

		if (params.filterByColor)
		{
			if (binaryImage.at<uchar>(cvRound(center.location.y), cvRound(center.location.x)) != params.blobColor)
				continue;
		}

		//compute blob radius
		{
			std::vector<double> dists;
			for (size_t pointIdx = 0; pointIdx < contours[contourIdx].size(); pointIdx++)
			{
				Point2d pt = contours[contourIdx][pointIdx];
				dists.push_back(norm(center.location - pt));
			}
			std::sort(dists.begin(), dists.end());
			center.radius = (dists[(dists.size() - 1) / 2] + dists[dists.size() / 2]) / 2.;
		}

		centers.push_back(center);


#ifdef DEBUG_BLOB_DETECTOR
		    circle( keypointsImage, center.location, 1, Scalar(0,0,255), 1 );
#endif
	}
#ifdef DEBUG_BLOB_DETECTOR
	  imshow("bk", keypointsImage );
	  waitKey();
#endif
}


void Multithreshould::detect(Mat& image, std::vector<cv::KeyPoint>& keypoints)
{
	//CV_INSTRUMENT_REGION()

	//TODO: support mask
	keypoints.clear();
	Mat grayscaleImage;
	if (image.channels() == 3)
		cvtColor(image, grayscaleImage, COLOR_BGR2GRAY);
	else
		grayscaleImage = image;

	if (grayscaleImage.type() != CV_8UC1) {
		CV_Error(Error::StsUnsupportedFormat, "Blob detector only supports 8-bit images!");
	}

	std::vector < std::vector<Center> > centers;
	for (double thresh = params.minThreshold; thresh < params.maxThreshold; thresh += params.thresholdStep)
	{
		Mat binarizedImage;
		threshold(grayscaleImage, binarizedImage, thresh, 255, THRESH_BINARY);

		std::vector < Center > curCenters;
		findBlobs(grayscaleImage, binarizedImage, curCenters);
		std::vector < std::vector<Center> > newCenters;
		for (size_t i = 0; i < curCenters.size(); i++)
		{
			bool isNew = true;
			for (size_t j = 0; j < centers.size(); j++)
			{
				double dist = norm(centers[j][centers[j].size() / 2].location - curCenters[i].location);
				isNew = dist >= params.minDistBetweenBlobs && dist >= centers[j][centers[j].size() / 2].radius && dist >= curCenters[i].radius;
				if (!isNew)
				{
					centers[j].push_back(curCenters[i]);

					size_t k = centers[j].size() - 1;
					while (k > 0 && centers[j][k].radius < centers[j][k - 1].radius)
					{
						centers[j][k] = centers[j][k - 1];
						k--;
					}
					centers[j][k] = curCenters[i];

					break;
				}
			}
			if (isNew)
				newCenters.push_back(std::vector<Center>(1, curCenters[i]));
		}
		std::copy(newCenters.begin(), newCenters.end(), std::back_inserter(centers));
	}

	for (size_t i = 0; i < centers.size(); i++)
	{
		if (centers[i].size() < params.minRepeatability)
			continue;
		Point2d sumPoint(0, 0);
		double normalizer = 0;
		for (size_t j = 0; j < centers[i].size(); j++)
		{
			sumPoint += centers[i][j].confidence * centers[i][j].location;
			normalizer += centers[i][j].confidence;
		}
		sumPoint *= (1. / normalizer);
		KeyPoint kpt(sumPoint, (float)(centers[i][centers[i].size() / 2].radius) * 2.0f);
		keypoints.push_back(kpt);
	}
}

Point2f Multithreshould::optimize(Mat& image)
{
	params = blob_black;
	vector<KeyPoint> pupil_can;
	detect(image, pupil_can);
#ifdef DEBUG_USER
	drawKeypoints(image, pupil_can, image);
#endif

	if (pupil_can.size() == 0) return Point2f(-1, -1);
 	KeyPoint pupil=seletBestPupil(pupil_can);
#ifdef DEBUG_USER
	drawMarker(image, pupil.pt, Scalar(0, 0, 255));
	imshow("PUpil", image);
	waitKey(10);
#endif


	params = blob_white;
#ifdef DEBUG_BLOB_DETECTOR
	cvtColor(image, image,CV_BGR2GRAY);
#endif
 	Mat inverse = inverse_gray(image);
	vector<KeyPoint> noisy;
	detect(inverse, noisy);
#ifdef DEBUG_USER
	drawKeypoints(inverse, noisy, inverse);
#endif
	double dx = 0, dy = 0, WhiteCount = 0, affectratio = 1;;
	for (int j = 0; j < noisy.size(); j++)
	{
		if (KeyPointDIS(noisy[j], pupil) < pupil.size / 1.8)
		{
			//circle(InputGrey, KeyPoints[j].pt, KeyPoints[j].size, Scalar(255, 255, 255));
			dx += noisy[j].pt.x*noisy[j].size*noisy[j].size;
			dy += noisy[j].pt.y*noisy[j].size*noisy[j].size;
			WhiteCount += noisy[j].size*noisy[j].size;
		}
	}
	pupil.pt.x = (pupil.pt.x*pupil.size*pupil.size + dx*affectratio) / (WhiteCount + pupil.size*pupil.size);
	pupil.pt.y = (pupil.pt.y*pupil.size*pupil.size + dy*affectratio) / (WhiteCount + pupil.size*pupil.size);
#ifdef DEBUG_USER
	imshow("PUpil", inverse);
	waitKey(10);
#endif
	return pupil.pt;
}

KeyPoint Multithreshould::seletBestPupil(vector<KeyPoint>& KeyPoints)
{
	struct KeyPointCMP
	{
		bool operator()(const KeyPoint& lhs, const KeyPoint& rhs)
		{
			return lhs.size < rhs.size;
		}
	};
	vector<KeyPoint>::iterator maxp = max_element(KeyPoints.begin(), KeyPoints.end(), KeyPointCMP());
	return *maxp;
}


Multithreshould::~Multithreshould()
{
}
