/*
 * solution.cpp
 *
 *  Created on: 20 déc. 2018
 *      Author: ajuven
 */




#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <stdio.h>

#include <string.h>

using namespace cv;
using namespace std;




Mat loadPicture(const char * const path)
{
	Mat output = imread(path, CV_LOAD_IMAGE_COLOR);

	if (output.data == NULL){
		std::cerr << "Invalid picture path : " << path << '\n';
		exit(EXIT_FAILURE);
	}

	return output;
}



Mat loadGrayPicture(const char * const path)
{
	Mat output = imread(path, CV_LOAD_IMAGE_GRAYSCALE);

	if (output.data == NULL){
		std::cerr << "Invalid picture path : " << path << '\n';
		exit(EXIT_FAILURE);
	}

	return output;
}



const char * name_from_path(const char * const str)
{
	const char * last_valid = str;

	for (const char * cursor = str ; *cursor != 0 ; cursor++){
		if (*cursor == '/' && *(cursor + 1) != 0){
			last_valid = cursor+1;
		}
	}

	return last_valid;
}



typedef struct  {
	int is_found;
	float centerX;
	float centerY;
	float radius;
} BallPosition;




#define MAX_AVERAGE_OTSU_PIXVAL 50.
#define MIN_CONTOUR_RADIUS 20.
#define MAX_CONTOUR_RADIUS 100.



#define DETAIL_SHOW(name)\
		do {\
			if (detail_please){\
				imshow(name, greyM);\
			}\
		} while (0);\


#define DETAIL_PLOT_SOL(center, radius)\
		do {\
			if (detail_please){\
				Mat __solM = loadPicture(imPath);\
				circle(__solM, Point(center.x, center.y), radius, Scalar(0,0,255));\
				imshow("Found solution", __solM);\
			}\
		} while (0);



#define DETAIL_WAIT()\
		do {\
			if (detail_please){\
				waitKey(0);\
				cvDestroyAllWindows();\
			}\
		} while (0);\



float shapeScore(Point2f center, int radius, vector<Point> contours ){
	float score = 0;
	for (unsigned j = 0 ; j < contours.size() ; j++){
		int norme_carree = abs(contours[j].x - center.x)*abs(contours[j].x - center.x) + abs(contours[j].y - center.y)*abs(contours[j].y - center.y);
		score += abs( sqrt( norme_carree ) - radius ) / radius;
	}
	return score / contours.size();
}

void printFail(const char * imPath, BallPosition ball)
{
	Mat img = imread(imPath);
	Point2f center;
	center.x = ball.centerX;
	center.y = ball.centerY;
	circle( img, center, (int)ball.radius, Scalar(255,0,0));
	imshow("fail : ", img);
	waitKey();
	return;
}

int cropIt(Mat image){
	Mat HSV, Hue;
	cvtColor(image, HSV, CV_BGR2HSV);
	Mat h_s_v[3];
	int value;
	int index = 0;
	split(HSV, h_s_v);
	for (int i = 0; i < h_s_v[2].rows; i++){
		value = 0;
		for (int j = 0; j < h_s_v[2].cols; j++){
			value += h_s_v[2].at<uchar>(i,j);
		}
		float mean = value/h_s_v[2].cols;
		if (mean > 120){
			index = i;
		}
	}

	if (index > image.cols / 2){ index = 0;}
	// std::cout << h_s_v[2].cols << index << h_s_v[2].cols-index << std::endl;
	// Mat crp = h_s_v[2](Rect(0, index, h_s_v[2].cols, h_s_v[2].rows-(index+2)));
	// imshow("cropped", crp);
	// waitKey();
	return index;
}






#define NO_BALL_FOUND ((BallPosition) {0, 0., 0., 0.})
#define MAX_SHAPE_SCORE 0.2

BallPosition ultimateSol(const char * const imPath, int detail_please = 0)
{
	Mat img = loadPicture(imPath);
	Mat greyMNoCroped = loadGrayPicture(imPath);


	int limitIndex = cropIt(img);
	Mat greyM;
	greyMNoCroped(Rect(0, limitIndex, greyMNoCroped.cols, greyMNoCroped.rows-limitIndex)).copyTo(greyM);

	DETAIL_SHOW("crop");

	fastNlMeansDenoising(greyM, greyM, 20., 7, 13);
	DETAIL_SHOW("denoise");


	threshold(greyM, greyM, 0, 255, THRESH_BINARY + THRESH_OTSU);
	DETAIL_SHOW("otsu");

	// improved method
	int morph_size = 4;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point(morph_size, morph_size));
	morphologyEx(greyM, greyM, CV_MOP_OPEN, element);
	DETAIL_SHOW("open");

	morph_size = 6;
	element = getStructuringElement(MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point(morph_size, morph_size));
	morphologyEx(greyM, greyM, CV_MOP_CLOSE, element);
	DETAIL_SHOW("close");

	Scalar m = mean(greyM);

	if (m.val[0] > MAX_AVERAGE_OTSU_PIXVAL){
		DETAIL_SHOW("MAX_AVERAGE_OTSU_PIXVAL reached");
		DETAIL_WAIT();
		return NO_BALL_FOUND;
	}


	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(greyM, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	DETAIL_SHOW("find countours");


	if (contours.size() == 0){
		DETAIL_SHOW("no countour found");
		DETAIL_WAIT();
		return NO_BALL_FOUND;
	}



	Point2f solCenter;
	float solRadius = 0;
	unsigned solIndex = 0;

	for (unsigned i = 0 ; i < contours.size() ; i++){

		Point2f center;
		float radius;
		float score;
		minEnclosingCircle(contours[i], center, radius);

		if (radius < MIN_CONTOUR_RADIUS || radius > MAX_CONTOUR_RADIUS){
			continue;
		}

		score = shapeScore(center, radius, contours[i]);

		if (score  >= MAX_SHAPE_SCORE){
			continue;
		}

		if (radius > solRadius){
			solCenter = center;
			solRadius = radius;
			solIndex = i;
		}
	}

	if (solRadius == 0){
		DETAIL_SHOW("no sol found");
		DETAIL_WAIT();
		return NO_BALL_FOUND;
	}


	drawContours(greyM, contours, solIndex, Scalar(255,255,255));
	DETAIL_SHOW("selected countour");


	solCenter.y += limitIndex;

	DETAIL_PLOT_SOL(solCenter, solRadius);
	DETAIL_WAIT();
	return (BallPosition) {1, solCenter.x, solCenter.y, solRadius};

}


