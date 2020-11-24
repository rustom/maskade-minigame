#pragma once

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include <iostream>
#include <opencv2/videoio.hpp>

#include "cppflow/cppflow.h"
#include "cppflow/model.h"
#include "cppflow/ops.h"
#include "opencv2/core.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/videoio.hpp"
using cv::FONT_HERSHEY_COMPLEX;
using cv::LINE_AA;
using cv::Mat;
using cv::Scalar;
using cv::VideoCapture;
using cv::waitKey;
using std::cout;
using std::endl;

namespace maskade { 

class MaskadeClassifier : public ci::app::App {
 public: 
  MaskadeClassifier();
  void Run();
  void setup() override;
  void mouseDown(ci::app::MouseEvent event) override;
  void update() override;
  void draw() override;
 private:
  
};

}

// void drawText(Mat& image) {
//   putText(image, "Hello OpenCV", cv::Point(20, 50), FONT_HERSHEY_COMPLEX,
//           1,                      // font face and scale
//           Scalar(255, 255, 255),  // white
//           1, LINE_AA);            // line thickness and type
// }