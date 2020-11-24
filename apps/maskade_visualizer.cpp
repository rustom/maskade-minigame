// The include statements below are intentionally long, but will be shortened
// when I flesh out more of the code

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
  cppflow::tensor input = cppflow::decode_jpeg(cppflow::read_file(
      std::string("/Users/rustomichhaporia/GitHub/Cinder/my-projects/"
                  "final-project-rustom-ichhaporia/assets/photo3.jpeg")));

  std::cout << input;
  // Cast the datatype of the input, expand dimensions, and change size to match
  // the image size of the model
  input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
  input = input / 255.f;
  input = cppflow::expand_dims(input, 0);
  std::cout << input.shape();
  auto il = {224, 224};
  input = cppflow::resize_bilinear(input, cppflow::tensor(il));
  std::cout << input.shape();

  // Load in the saved model built online with Google's Teachable Machines
  // project https://teachablemachine.withgoogle.com/train
  cppflow::model model(
      "/Users/rustomichhaporia/GitHub/Cinder/my-projects/"
      "final-project-rustom-ichhaporia/assets/converted_savedmodel/"
      "model.savedmodel");

  // Print list of possible operations on the Tensor model
  // std::vector<std::string> ops = model.get_operations();
  // for (auto item : ops) {
  //   std::cout << item << std::endl << std::endl;
  // }

  // Output the prediction from the model
  auto output = model(input);
  std::cout << output;

  // return 0;

  // The code below connects OpenCV binaries built locally to the laptop's
  // camera feed Some code is taken from online OpenCV examples for proof of
  // concept This can only be done in superuser mode on VS code

  int IMG_SIZE = 224;

  std::cout << output << std::endl;

  cout << "Built with OpenCV " << CV_VERSION << endl;
  cv::Mat image;
  VideoCapture capture;
  capture.open(0);
  if (capture.isOpened()) {
    cout << "Capture is opened" << endl;
    for (;;) {
      // for (size_t i = 0; i < 1; ++i) {
      capture >> image;
      // cv::flip(image, image, 1);
      // image = image(cv::Rect(540, 360, IMG_SIZE, IMG_SIZE));

      imshow("Sample", image);

      std::cout << image.size;
      image.convertTo(image, CV_32F);
      cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
      image /= 255.f;

      // Image dimensions
      int rows = image.rows;
      int cols = image.cols;
      int channels = image.channels();
      int total = image.total();

      std::cout << rows << ", " << cols << ", " << channels << ", " << total;

      // Assign to vector for 3 channel image
      // Souce: https://stackoverflow.com/a/56600115/2076973
      Mat flat = image.reshape(1, image.total() * channels);

      std::vector<float> img_data(IMG_SIZE * IMG_SIZE * 3);
      img_data = image.isContinuous()? flat : flat.clone();
      std::cout << std::endl;
      for(size_t i = 0; i < 20; ++i) {
        std::cout << img_data[i] << ",";
        
      }
      cppflow::tensor tensor(img_data, {1, rows, cols, channels});
      std::cout << tensor.dtype();
      // tensor = tensor/255.f;
      auto dims = {224, 224};
  tensor = cppflow::resize_bilinear(tensor, cppflow::tensor(dims));

      // auto il = {224, 224};
      // input = cppflow::resize_bilinear(tensor, cppflow::tensor(il));

      auto output_2 = model(tensor);
      
      std::cout << "This is the prediction" << output_2;

      // std::cout << image;
      // cvtColor(image, image, CV_RGB);
      // std::cout << (float*)image.data;
      // std::cout << image.

      // std::vector<float> array((float*)image.data, (float*)image.data + image.rows * image.cols);
      // std::cout << array.size() << std::endl;
 
      // std::cout << image.size;
      // //   image.reshape(0, std::vector<int>{224,224});
      // //   std::cout << image.size;
      // cv::resize(image, image, cv::Size(224, 224));
      // image.convertTo(image, CV_32F, 1.0/255.0);
      // std::cout << image.size;
      // std::cout << image;
      // int rows = image.rows;
      // int cols = image.cols;
      // int channels = image.channels();
      // int total = image.total();
      // Mat flat = image.reshape(1, image.total() * channels);
      // Mat flat = image;

      // std::vector<float> img_data;
      // img_data = image.isContinuous() ? flat : flat.clone();

      // Run and show predictions
      // cppflow::tensor tensor(img_data, {1, rows, cols, channels});
      // auto output_2 = model(image);
      // std::cout << "This is the prediction" << output_2;

      // // Get tensor with predictions
      // std::vector<float> predictions = prediction.Tensor::get_data<float>();
      // for(int i=0; i<predictions.size(); i++)
      //     std::cout<< std::to_string(predictions[i]) << std::endl;

      // cppflow::tensor ten = inp;

      if (image.empty())
        break;
      drawText(image);
      // imshow("Sample", image);
      if (waitKey(10) >= 0)
        break;
    }
  } else {
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

// The commented code below is the skeleton for the integration of the Cinder
// app with TensorFlow and OpenCV

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
