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
	imshow(imPath, greyM);

	imshow("initial", greyM);

	fastNlMeansDenoising(greyM, greyM, 20., 7, 13);
	imshow("nl_means", greyM);

	threshold(greyM, greyM, 0, 255, THRESH_BINARY + THRESH_OTSU);
	imshow("otsu", greyM);


	int morph_size = 4;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point(morph_size, morph_size));
	morphologyEx(greyM, greyM, CV_MOP_OPEN, element);
	imshow("morph", greyM);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(greyM, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	imshow("contours", greyM);

	Scalar greyColor = Scalar(255);
	drawContours(greyM, contours, 0, greyColor);
	imshow("first contour drawn", greyM);

	drawContours(greyM, contours, -1, greyColor);
	imshow("every contour drawn", greyM);


	Point2f center;
	float radius;

	minEnclosingCircle(contours[0], center, radius);

	cout << "center : " << center << "\n";
	cout << "radius : " << radius << "\n";

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

