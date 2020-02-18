#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/stitching.hpp" 
//#include "opencv2/matplotlibcpp.hpp"
//#include "misc.h"
using namespace cv;
using namespace std;
const float GOOD_MATCH_PERCENT = 0.05f;
/**code for fast+freak**/

using namespace cv::xfeatures2d;
Stitcher::Mode mode = Stitcher::PANORAMA;
/** @function main */
int main(int argc, char** argv)
{
	Mat img_1 = imread("keble_a.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_2 = imread("keble_b.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_1_color=imread("keble_a.jpg");
	Mat img_2_color = imread("keble_b.jpg");
	
	//imshow("img_1", img_1);
	//-- Step 1: Detect the keypoints using SIFT Detector
	Ptr<SIFT> detector = SIFT::create();
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);
	//-- Draw keypoints
	Mat img_keypoints_1; Mat img_keypoints_2;
	drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1),
		DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1),
		DrawMatchesFlags::DEFAULT);
	imwrite("keypoints1.jpg",img_keypoints_1);
	imwrite("keypoints2.jpg",img_keypoints_2);
	
	
	//DescriptorExtractor* extractor;
	//extractor = new SiftDescriptorExtractor();
	Ptr<FREAK> extractor = FREAK::create();

	Mat desc1, desc2;
	// Step-3: Compute feature descriptors for all points in keypoints_i
	extractor->compute(img_1, keypoints_1, desc1);
	extractor->compute(img_2, keypoints_2, desc2);
	//imshow("Keypoints 1", img_keypoints_1);
	//imshow("Keypoints 2", img_keypoints_2);
	// Match features.
	std::vector<DMatch> matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(desc1, desc2, matches, Mat());
	Mat immatch;
	drawMatches(img_1_color, keypoints_1, img_2_color, keypoints_2, matches, immatch);
	//imshow("matches1", immatch);

	//// Sort matches by score
	std::sort(matches.begin(), matches.end());

	//// Remove not so good matches
	const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
	matches.erase(matches.begin() + numGoodMatches, matches.end());


	//// Draw top matches
	Mat imMatches;
	drawMatches(img_1_color, keypoints_1, img_2_color, keypoints_2, matches, imMatches);
	//imwrite("matches.jpg", imMatches);
	imshow("match",imMatches);

	std::vector<Point2f> points1, points2;

	for (size_t i = 0; i < matches.size(); i++)
	{
		points1.push_back(keypoints_1[matches[i].queryIdx].pt);
		points2.push_back(keypoints_2[matches[i].trainIdx].pt);
	}
	Mat result, destination, descriptors_updated;
	vector<Point2f> fourPoint;
	vector<KeyPoint> keypoints_updated;
	Mat h;// (3, 3, CV_64FC1);
// Find homography
	h = findHomography(points1, points2, RANSAC);

	Mat C = (Mat_<double>(3, 3) << 1, 0, 300, 0, 1, 0, 0, 0, 1);
	//cout << "C = " << endl << " " << C << endl << endl;
	//result = Mat(Size(tam_x * 2, tam_y * 2), CV_8UC3, cv::Scalar(0, 0, 0));
	//cout << Htr * h;
	Mat result1;

	warpPerspective(img_1, result1, h, img_1.size(), INTER_LINEAR, BORDER_TRANSPARENT, 0);
	warpPerspective(img_1_color, result, (C * h), img_1_color.size(), INTER_LINEAR, BORDER_TRANSPARENT, 0);
	
	imshow("result", result);
	imwrite("warped.jpg", result);
	vector<Mat> imgs;
	imgs.push_back(result);
	imgs.push_back(img_2_color);
	Mat pano;
	Ptr<Stitcher> stitcher = Stitcher::create(mode);
	Stitcher::Status status = stitcher->stitch(imgs, pano);
	if (status != Stitcher::OK)
	{
		cout << "Can't stitch images, error code = " << int(status) << endl;
		return EXIT_FAILURE;
	}
//	imshow("pano", pano);
	

	//imshow("img_2", img_2);
	
	imwrite("stitched_image2.jpg", pano);
	waitKey(0);
	return 0;
}