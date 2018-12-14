#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>

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
			Vec3b hsv=HSV.at<Vec3b>(center.x + i*(radius + offset),center.y + j*(radius + offset));

			float H=hsv.val[0] * 2; //opencv range : 180
			float S=hsv.val[1]/255.*100;  //opencv range : 255
			float V=hsv.val[2]/255.*100;  //opencv range : 255

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



void process(const char * const imPath)
{
	Mat greyM = loadGrayPicture(imPath);
	//imshow(imPath, greyM);

	//imshow("initial", greyM);

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
	//imshow("contours", greyM);

	Scalar greyColor = Scalar(255);
	// drawContours(greyM, contours, 0, greyColor);
	//imshow("first contour drawn", greyM);

	// drawContours(greyM, contours, -1, greyColor);
	// imshow("every contour drawn", greyM);


	

	// usual threshold : [85;45;30] to [100;65;30] //
	Mat img = imread(imPath);
	int thresh_min_in[3] = {85,45,30};
	int thresh_max_in[3] = {100,65,30};
	int thresh_min_ext[3] = {85,45,30};
	int thresh_max_ext[3] = {100,65,30};

	int i = 0;
	Point2f center;
	float radius;
	float score_couleur = 0;
	while( score_couleur <= 0.2 && i < contours.size()){
		minEnclosingCircle(contours[i], center, radius);
		float score_couleur_in = color_scoring(img, center, radius, thresh_min_in, thresh_max_in, 4);
		float score_couleur_ext = color_scoring(img, center, radius, thresh_min_ext, thresh_max_ext, -4);
		score_couleur = score_couleur_in + score_couleur_ext / 2;
		i++;
	}
	// si le score n'est pas probant, recupere le contour le mieux classé par findContours
	if ( score_couleur < 0.2 ){
		minEnclosingCircle(contours[0], center, radius);
	}


	cout << "center : " << center << "\n";
	cout << "radius : " << radius << "\n";
	cout << "score : " << score_couleur << endl;
 
	Mat output = loadPicture(imPath);
	Scalar color = Scalar(0,0,255);
	circle(output, center, (int) radius, color);
	imshow("circle drawn", output);


	FILE * f = fopen("test.txt", "a+");

	if ((char) waitKey(0) == 's'){
		fprintf(f, "%s %f %f %f\n", name_from_path(imPath), center.x, center.y, radius);
	}

	fclose(f);
}



int main(int argc, char * argv[])
{

	if (argc != 2){
		cerr << "Usage : " << argv[0] << " imPath\n";
		exit(EXIT_FAILURE);
	}

	const char * const imPath = argv[1];

	process(imPath);

	return 0;
}

