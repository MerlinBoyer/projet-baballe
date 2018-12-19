/*
 * test_method.cpp
 *
 *  Created on: 14 déc. 2018
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


BallPosition method1(const char * const imPath)
{
	Mat greyM = loadGrayPicture(imPath);

	fastNlMeansDenoising(greyM, greyM, 20., 7, 13);

	threshold(greyM, greyM, 0, 255, THRESH_BINARY + THRESH_OTSU);

	int morph_size = 4;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point(morph_size, morph_size));
	morphologyEx(greyM, greyM, CV_MOP_OPEN, element);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(greyM, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	/*
	  for( int i = 0; i< contours.size(); i++ ) // iterate through each contour.
	      {
	       double a=contourArea( contours[i],false);  //  Find the area of contour
	       if(a>largest_area){
	       largest_area=a;
	       largest_contour_index=i;                //Store the index of largest contour
	       bounding_rect=boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
	       }

	      }
	 */

	Point2f center;
	float radius;
	minEnclosingCircle(contours[0], center, radius);

	return (BallPosition) {1, center.x, center.y, radius};
}


#define MAX_AVERAGE_OTSU_PIXVAL 50.

#define MIN_CONTOUR_RADIUS 10.
#define MAX_CONTOUR_RADIUS 150.


BallPosition selectLargestContour(const char * const imPath)
{
	Mat greyM = loadGrayPicture(imPath);

	fastNlMeansDenoising(greyM, greyM, 20., 7, 13);

	threshold(greyM, greyM, 0, 255, THRESH_BINARY + THRESH_OTSU);


	int morph_size = 4;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point(morph_size, morph_size));
	morphologyEx(greyM, greyM, CV_MOP_OPEN, element);

	Scalar m = mean(greyM);
	cout << m.val[0] << "\n";

	if (m.val[0] > MAX_AVERAGE_OTSU_PIXVAL){
		return (BallPosition) {0, 0., 0., 0.};
	}


	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(greyM, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	double largest_area = 0.;
	unsigned largest_contour_index = 0;

	for (unsigned i = 0; i< contours.size(); i++ ){
		double a = contourArea(contours[i], false);

		if (a > largest_area){
			largest_area = a;
			largest_contour_index=i;
		}
	}

	Point2f center;
	float radius;
	minEnclosingCircle(contours[largest_contour_index], center, radius);

	if (radius < MIN_CONTOUR_RADIUS || radius > MAX_CONTOUR_RADIUS){
		return (BallPosition) {0, 0., 0., 0};
	}


	return (BallPosition) {1, center.x, center.y, radius};
}







typedef BallPosition (*BallFinderMethod)(const char * const fileName);



float radiusPrecision(BallPosition & prediction, BallPosition & correct)
{
	float delta_r = prediction.radius - correct.radius;

	if (delta_r < 0){
		delta_r *= -1;
	}

	return delta_r / correct.radius;
}


float centerPrecision(BallPosition & prediction, BallPosition & correct)
{
	float deltaX = prediction.centerX - correct.centerX;
	float deltaY = prediction.centerY- correct.centerY;

	return sqrtf(deltaX * deltaX + deltaY * deltaY)/ correct.radius;
}


#define MAX_RADIUS_PRECISION 0.25
#define MAX_CENTER_PRECISION 0.25



void processScore(BallFinderMethod method, const char * const trainResults, const char * const testFolder)
{
	FILE * results = fopen(trainResults, "r");

	if (results == NULL){
		cerr << "Couldn't open " << trainResults << " file \n";
		exit(EXIT_FAILURE);
	}

	char file_name[512];
	size_t folerNameSize = strlen(testFolder);

	strcpy(file_name, testFolder);
	file_name[folerNameSize] = '/';

	BallPosition correctPos;

	unsigned totalNbTests = 0;
	unsigned nbTestsWithBall = 0;


	unsigned nbTreated = 0;
	unsigned nbTestsWithBallButNotFound = 0;
	unsigned nbTestsWithNoBallButFound = 0;

	unsigned nbPassed = 0;

	float radPrecSum = 0.;
	float centerPrecSum = 0.;
	float maxRadPrec = 0.;
	float maxCenterPrec = 0.;


	while (EOF != fscanf(results, "%s %d %f %f %f\n",
			file_name + folerNameSize + 1,
			&(correctPos.is_found), &(correctPos.centerX),
			&(correctPos.centerY), &(correctPos.radius))){

		totalNbTests++;

		if (correctPos.is_found){
			nbTestsWithBall++;
		}

		BallPosition predictedPos = method(file_name);

		if (!predictedPos.is_found){

			if (correctPos.is_found){
				nbTestsWithBallButNotFound++;
				cout << file_name << " !!!\n";
			}
			continue;

		} else if (!correctPos.is_found){
			nbTestsWithNoBallButFound++;
			cout << file_name << " !!!!!!\n";
			continue;
		}


		nbTreated++;

		float radPrec = radiusPrecision(predictedPos, correctPos);
		float centerPrec = centerPrecision(predictedPos, correctPos);

		if (maxRadPrec < radPrec){
			maxRadPrec = radPrec;
		}

		if (maxCenterPrec < centerPrec){
			maxCenterPrec = centerPrec;
		}

		if (radPrec > MAX_RADIUS_PRECISION || centerPrec > MAX_CENTER_PRECISION){
			cout << file_name << " " << radPrec << ", " << centerPrec << "\n";
		} else {
			nbPassed++;
		}

		radPrecSum += radPrec;
		centerPrecSum += centerPrec;
	}

	float avgRadPrec = nbTreated ? radPrecSum / nbTreated : 0.;
	float avgCenterPrec = nbTreated ? centerPrecSum / nbTreated : 0.;


	cout << "\n\n\tRESULTS\n\n";

	cout << "totalNbTests : " << totalNbTests << "\n\n";

	cout << "nbTestsWithBall : " << nbTestsWithBall << "\n";
	cout << "nbTestsWithBallButNotFound : " << nbTestsWithBallButNotFound <<"\n";
	cout << "nbTestsWithNoBallButFound : " << nbTestsWithNoBallButFound <<"\n\n";

	cout << "nbTreatedBallScores : " << nbTreated << "\n";
	cout << "nbScorePassed : " << nbPassed << "\n\n";

	cout << "avgRadPrec : " << avgRadPrec << "\n";
	cout << "avgCenterPrec : " << avgCenterPrec << "\n\n";

	cout << "maxRadPrec : " << maxRadPrec << "\n";
	cout << "maxCenterPrec : " << maxCenterPrec << "\n";

	fclose(results);
}


int main(int argc, char * argv[])
{
	if (argc != 3){
		cerr << "Usage : " << argv[0] << " trainResults testFolder\n";
		exit(EXIT_FAILURE);
	}

	processScore(selectLargestContour, argv[1], argv[2]);

	return 0;
}
