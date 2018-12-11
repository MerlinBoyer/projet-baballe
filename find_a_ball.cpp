#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>

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



void process(const char * const imPath)
{
	Mat m = loadPicture(imPath);

	imshow(imPath, m);


	Mat greyM = loadGrayPicture(imPath);
	imshow(imPath, greyM);

	Mat thres_m;
	/*double found_thred =*/ threshold(greyM, thres_m, 0, 255, THRESH_BINARY + THRESH_OTSU);
	imshow("otsu-th-ocv.png", thres_m);

	waitKey(0);
}








int main(int argc, char * argv[])
{

	if (argc != 2){
		cerr << "Usage : " << argv[0] << "imPath\n";
		exit(EXIT_FAILURE);
	}

	const char * const imPath = argv[1];

	process(imPath);

	return 0;
}

