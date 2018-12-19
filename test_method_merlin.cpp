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

	//fastNlMeansDenoising(greyM, greyM, 20., 7, 13);

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

/*
* return a score between 0 and 1 depending on nb of neighbors corresponding to threshold
* usual threshold : [85;45;30] to [100;65;30]
*/
float color_scoring(Mat img, Point2f center, int radius, int hsv_min[3], int hsv_max[3], int offset){
	Mat HSV;
	cvtColor(img, HSV, COLOR_BGR2HSV);
	float score = 0;
	
	for (int i = -1; i<2; i+=2){
		for (int j = -1; j<2; j+=2){

			// check pixels outside the ball
			if ( center.x + i*(radius + offset) > HSV.rows || center.y + j*(radius + offset) > HSV.cols ){
				return 0;
			}
			Vec3b hsv=HSV.at<Vec3b>(center.x + i*(radius + offset),center.y + j*(radius + offset));

			float H=hsv.val[0] * 2; //opencv range : 180
			float S=hsv.val[1]/255.*100;  //opencv range : 255
			float V=hsv.val[2]/255.*100;  //opencv range : 255

			//cout << "H : " << H << " S : " << S << " V : " << V << endl;

			if ( hsv_min[0] < H && H < hsv_max[0]){
				score++;
			}
			if( hsv_min[1] < S && S < hsv_max[1]){
				score++;
			}
			if ( hsv_min[2] < V && V < hsv_max[2]){
				score++;
			}
		}
	}

	return score / 12;
}

#define THRESH_COLOR_SCORE 0.5

BallPosition selectLargestContour(const char * const imPath)
{
	Mat img = imread(imPath);
	Mat greyM = loadGrayPicture(imPath);

	blur(greyM, greyM, Size(3,3));
	//imshow("blur", greyM);

	threshold(greyM, greyM, 0, 255, THRESH_BINARY + THRESH_OTSU);
	//imshow("otsu", greyM);

	int morph_size; Mat element; Mat temp;

	// improved method
	morph_size = 4;
	element = getStructuringElement(MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point(morph_size, morph_size));
	morphologyEx(greyM, greyM, CV_MOP_OPEN, element);
	//imshow("morph open 1", greyM);
	morph_size = 6;
	element = getStructuringElement(MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point(morph_size, morph_size));
	morphologyEx(greyM, greyM, CV_MOP_CLOSE, element);
	//imshow("morph close 2", greyM);

	

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(greyM, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);


	// usual threshold : [85;45;30] to [100;65;30] //
	
	int thresh_min_in[3] = {30,45,30};
	int thresh_max_in[3] = {40,80,30};
	int thresh_min_ext[3] = {85,45,30};
	int thresh_max_ext[3] = {100,65,30};

	int i = 0;int max = 0;
	Point2f center;
	float radius;
	float score_couleur = 0;
	
	while( score_couleur <= THRESH_COLOR_SCORE && i < contours.size()){
	 	minEnclosingCircle(contours[i], center, radius);
	 	float score_couleur_in = color_scoring(img, center, radius, thresh_min_in, thresh_max_in, 4);
	 	float score_couleur_ext = color_scoring(img, center, radius, thresh_min_ext, thresh_max_ext, -4);
	 	float score_tot = score_couleur_in + score_couleur_ext / 2;
	 	max = (score_tot > score_couleur) ? i : max;
	 	score_couleur = score_tot;
	 	i++;
	}
	cout << endl;
	// si le score n'est pas probant, recupere le contour le mieux classé par findContours
	if ( score_couleur <= THRESH_COLOR_SCORE ){
		return (BallPosition) {0, center.x, center.y, radius};
	}
	

	// center : [17.5, 240.5]  radius : 6.71478
	// cout << "center : " << center << "\n";
	// cout << "radius : " << radius << "\n";
	// cout << "score : " << score_couleur << endl;
 
	// Mat output = loadPicture(imPath);
	Scalar color = Scalar(0,0,255);
	circle(img, center, (int) radius, color);

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
