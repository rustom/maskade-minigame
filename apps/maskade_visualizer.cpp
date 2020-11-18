// The include statements below are intentionally long, but will be shortened when
// I flesh out more of the code

// #include "cinder/app/App.h"
// #include "cinder/app/RendererGl.h"
// #include "cinder/gl/gl.h"

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

void drawText(Mat& image);

int main() {
  // Read in a sample image (hopefully, this will later be from the camera feed)
  auto input = cppflow::decode_jpeg(cppflow::read_file(
      std::string("/Users/rustomichhaporia/GitHub/Cinder/my-projects/"
                  "final-project-rustom-ichhaporia/assets/photo.jpeg")));

  // Cast the datatype of the input, expand dimensions, and change size to match the image size of the model
  input = cppflow::cast(input, TF_UINT8, TF_INT32);
  input = cppflow::expand_dims(input, 0);
  std::cout << input.shape();
  auto il = {224, 224};
  input = cppflow::resize_bilinear(input, cppflow::tensor(il));
  std::cout << input.shape();

  // Load in the saved model built online with Google's Teachable Machines project
  // https://teachablemachine.withgoogle.com/train
  cppflow::model model(
      "/Users/rustomichhaporia/GitHub/Cinder/my-projects/"
      "final-project-rustom-ichhaporia/assets/converted_savedmodel/"
      "model.savedmodel");

  // Print list of possible operations on the Tensor model
  std::vector<std::string> ops = model.get_operations();
  for (auto item : ops) {
    std::cout << item << std::endl << std::endl;
  }

  // Output the prediction from the model
  auto output = model(input);
  std::cout << output;

  // The code below connects OpenCV binaries built locally to the laptop's camera feed
  // Some code is taken from online OpenCV examples for proof of concept
  // This can only be done in superuser mode on VS code

  std::cout << output << std::endl;

  cout << "Built with OpenCV " << CV_VERSION << endl;
  cv::Mat image;
  VideoCapture capture;
  capture.open(0);
  if(capture.isOpened())
  {
      cout << "Capture is opened" << endl;
      for(;;)
      {
          capture >> image;
          if(image.empty())
              break;
          drawText(image);
          imshow("Sample", image);
          if(waitKey(10) >= 0)
              break;
      }
  }
  else
  {
      cout << "No capture" << endl;
      image = Mat::zeros(480, 640, CV_8UC1);
      drawText(image);
      imshow("Sample", image);
      waitKey(0);
  }
  return 0;
}

void drawText(Mat& image) {
  putText(image, "Hello OpenCV", cv::Point(20, 50), FONT_HERSHEY_COMPLEX,
          1,                      // font face and scale
          Scalar(255, 255, 255),  // white
          1, LINE_AA);            // line thickness and type
}

// The commented code below is the skeleton for the integration of the Cinder app with 
// TensorFlow and OpenCV

// using namespace ci;
// using namespace ci::app;
// using namespace std;

// class finalprojectApp : public App {
//   public:
// 	void setup() override;
// 	void mouseDown( MouseEvent event ) override;
// 	void update() override;
// 	void draw() override;

// };

// void finalprojectApp::setup()
// {
// }

// void finalprojectApp::mouseDown( MouseEvent event )
// {
// }

// void finalprojectApp::update()
// {
// }

// void finalprojectApp::draw()
// {
// 	gl::clear( Color( 0, 0, 0 ) );
// }

// CINDER_APP( finalprojectApp, RendererGl )
