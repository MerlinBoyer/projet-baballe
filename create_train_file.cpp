
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

	Point2f center;
	float radius;
	minEnclosingCircle(contours[0], center, radius);

	return (BallPosition) {1, center.x, center.y, radius};


}







void process(const char * const imPath, const char * const trainFile)
{
	FILE * f = fopen(trainFile, "a+");

	BallPosition res = method1(imPath);
	Mat display = loadPicture(imPath);
	Scalar foundColor = Scalar(0,0,255);
	Mat im = loadPicture(imPath);

	float speed = 1.;

	while (1){

		Mat display;
		im.copyTo(display);

		circle(display, Point(res.centerX, res.centerY), (int) res.radius, foundColor);
		imshow("Result", display);

		char c = waitKey(0);


		switch (c){

		case 's':
			fprintf(f, "%s %d %f %f %f\n", name_from_path(imPath), res.is_found, res.centerX, res.centerY, res.radius);
			fclose(f);
			return;

		case 'n':
			fprintf(f, "%s 0 0. 0. 0.\n", name_from_path(imPath));
			fclose(f);
			return;

		case 'k':
			res.centerX -= speed;
			break;

		case 'm':
			res.centerX += speed;
			break;

		case 'o':
			res.centerY -= speed;
			break;

		case 'l':
			res.centerY += speed;
			break;

		case 'y':
			res.radius += speed;
			break;

		case 'u':
			res.radius -= speed;
			break;

		case '+':
			speed *= 2;
			break;

		case '-':
			if (speed > 1){
				speed /= 2;
			}
			break;

		default:
			break;

		}


	}
}


int main(int argc, char * argv[])
{
	if (argc != 3){
		cerr << "Usage : " << argv[0] << " imPath outputTrainFile\n";
		exit(EXIT_FAILURE);
	}

	process(argv[1], argv[2]);

	return 0;
}
