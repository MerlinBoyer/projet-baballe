#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <stdio.h>

#include <string.h>

using namespace cv;
using namespace std;






/////////// usefull fct

/*
*  Load a picture in RGB
*/
Mat loadPicture(const char * const path)
{
	Mat output = imread(path, CV_LOAD_IMAGE_COLOR);

	if (output.data == NULL){
		std::cerr << "Invalid picture path : " << path << '\n';
		exit(EXIT_FAILURE);
	}

	return output;
}

/*
*  Load a picture in gray
*/
Mat loadGrayPicture(const char * const path)
{
	Mat output = imread(path, CV_LOAD_IMAGE_GRAYSCALE);

	if (output.data == NULL){
		std::cerr << "Invalid picture path : " << path << '\n';
		exit(EXIT_FAILURE);
	}

	return output;
}


/*
*  return last name in a path.
*  ex : ''/net/enseirb' => 'enseirb'
*/
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


/*
*  Struc containing all ball tracking result
*/
typedef struct  {
	int is_found;
	float centerX;
	float centerY;
	float radius;
} BallPosition;


/*
*  Threshold used in the algorithm
*/
#define MAX_AVERAGE_OTSU_PIXVAL 60.
#define MIN_CONTOUR_RADIUS 10.
#define MAX_CONTOUR_RADIUS 150.

#define MAX_RADIUS_PRECISION 0.25
#define MAX_CENTER_PRECISION 0.25


/*
*  draw circle corresponding to BallPosition arg and print image. This function
*  is called when main algorithm has failed to check where was the ball found
*  on the image.
*/
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


/*
*  Return a score based on correponding rate between a shape (OpenCV contour)
*  and its minEnclosingCircle. The lowest score match to the roundest contour.
*/
float shapeScore(Point2f center, int radius, vector<Point> contours ){
	float score = 0;
	for (unsigned j = 0 ; j < contours.size() ; j++){
		int norme_carree = abs(contours[j].x - center.x)*abs(contours[j].x - center.x) + abs(contours[j].y - center.y)*abs(contours[j].y - center.y);
		score += abs( sqrt( norme_carree ) - radius ) / radius;
	}
	return score / contours.size();
}


/*
*  check luminosity average for each row and return index (= image row number)
*  if its luminosity is over a threshold. The aim is to crop bright top part (if
*  there is one) because it can not be the field.
*/
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
	return index;
}







/*
*  Main algorithm
*/
BallPosition mostRound(const char * const imPath, int verbose = 0)
{
	Mat img = imread(imPath);
	Mat greyM = loadGrayPicture(imPath);
	int limitIndex = cropIt(img);
	if (verbose){
		Mat crp = greyM(Rect(0, limitIndex, greyM.cols, greyM.rows-(limitIndex+2)));
		imshow("cropped", crp);
	}

	fastNlMeansDenoising(greyM, greyM, 20., 7, 13);

	threshold(greyM, greyM, 0, 255, THRESH_BINARY + THRESH_OTSU);


	// improved method
	int morph_size = 4;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point(morph_size, morph_size));
	morphologyEx(greyM, greyM, CV_MOP_OPEN, element);
	if ( verbose )
	  imshow("morph open 1", greyM);
	morph_size = 6;
	element = getStructuringElement(MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point(morph_size, morph_size));
	morphologyEx(greyM, greyM, CV_MOP_CLOSE, element);
	if ( verbose )
	  imshow("morph close 2", greyM);

	Scalar m = mean(greyM);

	if (m.val[0] > MAX_AVERAGE_OTSU_PIXVAL){
		cout << " /!/ Otsu overflow error " << endl;
		if ( verbose ){
			waitKey();
		}
		return (BallPosition) {0, 0., 0., 0.};
	}


	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(greyM, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	if (contours.size() == 0){
		cout << " /!/ no contours found " << endl;
		if ( verbose )
			waitKey();
		return  (BallPosition) {0, 0., 0., 0.};
	}

	Point2f centerSol;
	float radiusSol = 0;
	float best_shape_score = 1;

	for (unsigned i = 0 ; i < contours.size() ; i++ ){

		Point2f center;
		float radius;
		minEnclosingCircle(contours[i], center, radius);

		if (radius < MIN_CONTOUR_RADIUS || radius > MAX_CONTOUR_RADIUS){
			continue;
		}
		if ( center.y < limitIndex ){
			continue;
		}

		float score = shapeScore( center, radius, contours[i]);

		if (score < best_shape_score ){
			best_shape_score = score;
			radiusSol = radius;
			centerSol = center;
			// cout << "new best score score : " << score << endl;
		}
	}

	if (radiusSol == 0){
		if ( verbose )
			waitKey();
		return  (BallPosition) {0, 0., 0., 0.};
	}
	if (best_shape_score >= 0.2){
		cout << " /!/ shape score not high enough : " << best_shape_score << endl;
		if ( verbose ){
			Scalar color = Scalar(0,0,255);
		  circle(img, centerSol, radiusSol, color);
		  imshow("result", img);
		  waitKey();
		}
		return  (BallPosition) {0, 0., 0., 0.};
	}

	cout << "found with shape score : " << best_shape_score << ", crop index : " << limitIndex << ", Otsu tr : " << m.val[0] << endl;

	Scalar color = Scalar(0,0,255);
	circle(img, centerSol, radiusSol, color);
	if (verbose){
		imshow("result", img);
		waitKey();
	}

	return (BallPosition) {1, centerSol.x, centerSol.y, radiusSol};
}















///////     test zone      /////////////////


typedef BallPosition (*BallFinderMethod)(const char * const fileName, int verbose);


/*
*  return the error between radius found and reference radius
*/
float radiusPrecision(BallPosition & prediction, BallPosition & correct)
{
	float delta_r = prediction.radius - correct.radius;

	if (delta_r < 0){
		delta_r *= -1;
	}

	return delta_r / correct.radius;
}

/*
*  return the error between coordinates found and reference coordinates
*/
float centerPrecision(BallPosition & prediction, BallPosition & correct)
{
	float deltaX = prediction.centerX - correct.centerX;
	float deltaY = prediction.centerY- correct.centerY;

	return sqrtf(deltaX * deltaX + deltaY * deltaY)/ correct.radius;
}




////////////////////////    test algorithm     ///////////////////

void processScore(BallFinderMethod method, const char * const trainResults, const char * const testFolder, int verbose)
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
				cout << file_name << " False Negative\n";
				method(file_name, verbose);
			}
			continue;

		} else if (!correctPos.is_found){
			nbTestsWithNoBallButFound++;
			cout << file_name << " FALSE POSITIVE !!!!!!\n";
			method(file_name, verbose);
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
			method(file_name, verbose);
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
	cout << "NbFalseNegatives : " << nbTestsWithBallButNotFound <<"\n";
	cout << "NbFalsePositives : " << nbTestsWithNoBallButFound <<"\n\n";

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
	if (argc != 4){
		cerr << "Usage : " << argv[0] << " trainResults testFolder verbose(=0 or 1)\n";
		exit(EXIT_FAILURE);
	}

	processScore(mostRound, argv[1], argv[2], atoi(argv[3]));

	return 0;
}
