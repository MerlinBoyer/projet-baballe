#include <iostream>
#include <cstdlib>


#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

enum {SQUARE, DIAMOND, DISK, LINE_V, DIAG_R, LINE_H, DIAG_L, CROSS, PLUS};

void 
process(const int shape, const int halfsize, const char* imdname)
{
  const int size = (2*halfsize + 1);
  Mat img_shape = Mat::zeros(size, size, CV_8UC1);
  
  if( shape == SQUARE ){ 
    for( int i = 0 ; i < size ; i++) {
      for( int j = 0 ; j < size ; j++) {
        img_shape.at<uchar>(i,j) = 255; 
      }
    }
  }

  else if( shape == DIAMOND ){
    for( int i = 0 ; i < size ; i++) {
      for( int j = 0 ; j < size ; j++) {
        if (abs(halfsize - i) + abs(halfsize - j) <= halfsize){
          img_shape.at<uchar>(i,j) = 255; 
        }
      }
    }
  }

  else if( shape == DISK ){
    for( int i = 0 ; i < size ; i++) {
      for( int j = 0 ; j < size ; j++) {
        if (abs(halfsize - i)*abs(halfsize - i) + abs(halfsize - j)*abs(halfsize - j) <= halfsize*halfsize){
          img_shape.at<uchar>(i,j) = 255; 
        }
      }
    }
  }

  else if( shape == LINE_V ){
    for( int i = 0 ; i < size ; i++) {
      for( int j = 0 ; j < size ; j++) {
        if (j == halfsize){
          img_shape.at<uchar>(i,j) = 255; 
        }
      }
    }
  }
  
  else if( shape == DIAG_R ){
    for( int i = 0 ; i < size ; i++) {
      for( int j = 0 ; j < size ; j++) {
        if (i == j){
          img_shape.at<uchar>(i,j) = 255; 
        }
      }
    }
  }

  else if( shape == LINE_H ){
    for( int i = 0 ; i < size ; i++) {
      for( int j = 0 ; j < size ; j++) {
        if (i == halfsize){
          img_shape.at<uchar>(i,j) = 255; 
        }
      }
    }
  }

  else if( shape == DIAG_L ){
    for( int i = 0 ; i < size ; i++) {
      for( int j = 0 ; j < size ; j++) {
        if ( i+j == size-1){
          img_shape.at<uchar>(i,j) = 255; 
        }
      }
    }
  }

  else if( shape == CROSS ){
    for( int i = 0 ; i < size ; i++) {
      for( int j = 0 ; j < size ; j++) {
        if (i+j == size-1 || i == j){
          img_shape.at<uchar>(i,j) = 255; 
        }
      }
    }
  }

  else if( shape == PLUS ){
    for( int i = 0 ; i < size ; i++) {
      for( int j = 0 ; j < size ; j++) {
        if (j == halfsize || i == halfsize){
          img_shape.at<uchar>(i,j) = 255; 
        }
      }
    }
  }

  imwrite(imdname, img_shape);
}

void 
usage (const char *s)
{
  std::cerr<<"Usage: "<<s<<" shape halfsize se-name"<<endl;
  exit(EXIT_FAILURE);
}

#define param 3
int 
main( int argc, char* argv[] )
{
  if(argc != (param+1))
    usage(argv[0]);
  process(atoi(argv[1]), atoi(argv[2]), argv[3]);
  return EXIT_SUCCESS;  
}

