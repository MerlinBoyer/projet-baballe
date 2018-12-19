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




typedef BallPosition (*BallFinderMethod)(const char * const fileName, int detail_please);



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


/*
void printStatus(const char * const fileName, BallPosition & correctPos, BallPosition & proposedPos)
{
	cout << "FAILED FOR : " << fileName << "\n";
	cout << "\tRadius : " << proposedPos.radius << ", expected " << correctPos << "\n";

}*/

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

		BallPosition predictedPos = method(file_name, 0);

		if (!predictedPos.is_found){

			if (correctPos.is_found){
				nbTestsWithBallButNotFound++;
				cout << file_name << " !!!\n";
				method(file_name, 1);
			}

			continue;

		} else if (!correctPos.is_found){
			nbTestsWithNoBallButFound++;
			cout << file_name << " FALSE POSITIVE !!!!!!\n";
			method(file_name, 1);
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
			method(file_name, 1);
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







#define MAX_AVERAGE_OTSU_PIXVAL 50.
#define MIN_CONTOUR_RADIUS 10.
#define MAX_CONTOUR_RADIUS 100.


BallPosition selectFirstContour(const char * const imPath, int detail_please)
{
	Mat greyM = loadGrayPicture(imPath);

	fastNlMeansDenoising(greyM, greyM, 20., 7, 13);

	threshold(greyM, greyM, 0, 255, THRESH_BINARY + THRESH_OTSU);


	int morph_size = 4;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point(morph_size, morph_size));
	morphologyEx(greyM, greyM, CV_MOP_OPEN, element);
	morphologyEx(greyM, greyM, CV_MOP_CLOSE, element);



	Scalar m = mean(greyM);

	if (m.val[0] > MAX_AVERAGE_OTSU_PIXVAL){
		return (BallPosition) {0, 0., 0., 0.};
	}


	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(greyM, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	Point2f center;
	float radius;
	minEnclosingCircle(contours[0], center, radius);

	if (radius < MIN_CONTOUR_RADIUS || radius > MAX_CONTOUR_RADIUS){
		return (BallPosition) {0, 0., 0., 0.};
	}


	return (BallPosition) {1, center.x, center.y, radius};
}



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


BallPosition selectLargestContour(const char * const imPath, int detail_please)
{
	Mat greyM = loadGrayPicture(imPath);

	fastNlMeansDenoising(greyM, greyM, 20., 7, 13);
	DETAIL_SHOW("nlmean");


	threshold(greyM, greyM, 0, 255, THRESH_BINARY + THRESH_OTSU);
	DETAIL_SHOW("otsu");


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
		DETAIL_SHOW("stopped by MAX_AVERAGE_OTSU_PIXVAL");
		DETAIL_WAIT();
		return (BallPosition) {0, 0., 0., 0.};
	}


	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(greyM, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	DETAIL_SHOW("findContours");

	double largest_area = 0.;
	unsigned largest_contour_index = 0;

	for (unsigned i = 0 ; i< contours.size() ; i++ ){

		double a = contourArea(contours[i], false);

		if (a > largest_area){
			largest_area = a;
			largest_contour_index=i;
		}
	}

	if (largest_area == 0){
		DETAIL_SHOW("No largest area found");
		DETAIL_WAIT();
		return (BallPosition) {0, 0., 0., 0.};
	}

	Point2f center;
	float radius;
	minEnclosingCircle(contours[largest_contour_index], center, radius);

	if (radius < MIN_CONTOUR_RADIUS || radius > MAX_CONTOUR_RADIUS){
		DETAIL_SHOW("stopped by MIN_MAX_CONTOUR_RADIUS");
		DETAIL_WAIT();
		return (BallPosition) {0, 0., 0., 0.};
	}

	DETAIL_SHOW("Solution returned");
	DETAIL_WAIT();
	return (BallPosition) {1, center.x, center.y, radius};
}





BallPosition InBounds(const char * const imPath, int detail_please)
{
	Mat greyM = loadGrayPicture(imPath);

	fastNlMeansDenoising(greyM, greyM, 20., 7, 13);

	threshold(greyM, greyM, 0, 255, THRESH_BINARY + THRESH_OTSU);
	DETAIL_SHOW("otsu");


	int morph_size = 6;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point(morph_size, morph_size));
	morphologyEx(greyM, greyM, CV_MOP_OPEN, element);
	DETAIL_SHOW("open");


	morph_size = 8;
	element = getStructuringElement(MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point(morph_size, morph_size));
	morphologyEx(greyM, greyM, CV_MOP_CLOSE, element);
	DETAIL_SHOW("close");



	Scalar m = mean(greyM);

	if (m.val[0] > MAX_AVERAGE_OTSU_PIXVAL){
		DETAIL_SHOW("stopped by MAX_AVERAGE_OTSU_PIXVAL");
		DETAIL_WAIT();

		return (BallPosition) {0, 0., 0., 0.};
	}


	/*
	Scalar m = mean(greyM);

	if (m.val[0] > MAX_AVERAGE_OTSU_PIXVAL){
		return (BallPosition) {0, 0., 0., 0.};
	}
	 */

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(greyM, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	DETAIL_SHOW("find contour");

	if (contours.size() == 0){
		DETAIL_SHOW("No contour found");
		DETAIL_WAIT();
		return  (BallPosition) {0, 0., 0., 0.};
	}

	Point2f centerSol;
	float radiusSol = 0;

	for (unsigned i = 0 ; i < contours.size() ; i++ ){

		Point2f center;
		float radius;
		minEnclosingCircle(contours[i], center, radius);

		if (radius < MIN_CONTOUR_RADIUS || radius > MAX_CONTOUR_RADIUS){
			continue;
		}

		if (radius > radiusSol){
			radiusSol = radius;
			centerSol = center;
		}
	}

	if (radiusSol == 0){
		DETAIL_SHOW("no solution found");
		DETAIL_WAIT();
		return  (BallPosition) {0, 0., 0., 0.};
	}

	DETAIL_PLOT_SOL(centerSol, radiusSol);
	DETAIL_WAIT();
	return (BallPosition) {1, centerSol.x, centerSol.y, radiusSol};
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
