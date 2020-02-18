#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/stitching.hpp" 
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
	Mat img_1 = imread("img1.png", 0);
	Mat img_2 = imread("img2.png", 0);
	Mat img_1_color = imread("img1.png");
	Mat img_2_color = imread("img2.png");

	vector<Mat> various_images;



	if (!img_1.data)
	{
		cout << "image not found" << endl;
	}
	//imshow("img_1", img_1);
	//imshow("img_2", img_2);
	Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);
	//-- Draw keypoints
	Mat img_keypoints_1; Mat img_keypoints_2;
	drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1),
		DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1),
		DrawMatchesFlags::DEFAULT);
	// Step-1: Create object for feature descriptor
	Ptr<FREAK> desc_comp = FREAK::create();
	// Step-2: Create destination array for descriptors

	Mat desc1, desc2;
	// Step-3: Compute feature descriptors for all points in keypoints_i
	desc_comp->compute(img_1, keypoints_1, desc1);
	desc_comp->compute(img_2, keypoints_2, desc2);
	//imwrite("Keypoints1.jpg", img_keypoints_1);
	//imwrite("Keypoints2.jpg", img_keypoints_2);
	// Match features.
	std::vector<DMatch> matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(desc1, desc2, matches, Mat());
	Mat immatch;
	drawMatches(img_1_color, keypoints_1, img_2_color, keypoints_2, matches, immatch);
	
	//imshow("matches", immatch);



	//// Sort matches by score
	std::sort(matches.begin(), matches.end());

	//// Remove not so good matches
	const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
	matches.erase(matches.begin() + numGoodMatches, matches.end());


	//// Draw top matches
	Mat imMatches;
	drawMatches(img_1_color, keypoints_1, img_2_color, keypoints_2, matches, imMatches);
	
	imshow("match",imMatches);
	//imwrite("matches.jpg", imMatches);

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

	Mat C = (Mat_<double>(3, 3) << 1, 0, 550, 0, 1, 0, 0, 0, 1);
	//cout << "C = " << endl << " " << C << endl << endl;
	//result = Mat(Size(tam_x * 2, tam_y * 2), CV_8UC3, cv::Scalar(0, 0, 0));
	//cout << Htr * h;
	Mat result1;

	warpPerspective(img_1, result1, h, img_1.size(), INTER_LINEAR, BORDER_TRANSPARENT, 0);
	warpPerspective(img_1_color, result, (C * h), img_2_color.size(), INTER_LINEAR, BORDER_TRANSPARENT, 0);

	//imwrite("warped.jpg",result);
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
	imshow("pano", pano);
	
	imwrite("stitched_image.jpg", pano);
	waitKey();
	return 0;
}
