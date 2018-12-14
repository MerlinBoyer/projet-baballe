#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <math.h>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void 
process(const char* imsname, const char* imdname)
{
    Mat image =  imread(imsname);
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
    std::cout << h_s_v[2].cols << index << h_s_v[2].cols-index << std::endl;
    Mat crp = h_s_v[2](Rect(0, index, h_s_v[2].cols, h_s_v[2].rows-(index+2)));
    imshow("H", crp);
    //merge(h_s_v, 3, HSV);
    //cvtColor(HSV, image, CV_HSV2BGR);
    imwrite(imdname, h_s_v[2]);
    waitKey(0);
}

void 
usage (const char *s)
{
  std::cerr<<"Usage: "<<s<<" imsname imdname\n"<<std::endl;
  exit(EXIT_FAILURE);
}

#define param 2
int 
main( int argc, char* argv[] )
{
  if(argc != (param+1))
    usage(argv[0]);

  FILE *fichier = NULL;
  fichier = fopen(argv[1],  "r");
 
  if (fichier == NULL){
      std::cerr<<"Impossible d'ouvrir le fichier !"<<std::endl;
      return EXIT_FAILURE;
  }
  process(argv[1], argv[2]);
  return EXIT_SUCCESS;  
}
