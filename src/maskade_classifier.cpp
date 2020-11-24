#include "maskade_classifier.hpp"

namespace maskade {

MaskadeClassifier::MaskadeClassifier() {
  
}

void MaskadeClassifier::Run() {
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

      std::cout << image.size;
      image.convertTo(image, CV_32F);
      cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
      image /= 255.f;

      // Image dimensions
      int rows = image.rows;
      int cols = image.cols;
      int channels = image.channels();
      int total = image.total();

      // Assign to vector for 3 channel image
      // Souce: https://stackoverflow.com/a/56600115/2076973
      Mat flat = image.reshape(1, image.total() * channels);

      std::vector<float> img_data(IMG_SIZE * IMG_SIZE * 3);
      img_data = image.isContinuous() ? flat : flat.clone();
      cppflow::tensor tensor(img_data, {1, rows, cols, channels});
      std::cout << tensor.dtype();
      // tensor = tensor/255.f;
      auto dims = {224, 224};
      tensor = cppflow::resize_bilinear(tensor, cppflow::tensor(dims));

      auto output_2 = model(tensor);

      auto argmax = cppflow::arg_max(output_2, 1).get_data<float>();

      if (argmax[0] == 0) {
        putText(image, "PUT ON YOUR MASK", cv::Point(20, 50),
                FONT_HERSHEY_COMPLEX,
                1,                      // font face and scale
                Scalar(255, 255, 255),  // white
                1, LINE_AA);            // line thickness and type
      } else {
        putText(image, "THANKS FOR WEARING YOUR MASK!", cv::Point(20, 50),
                FONT_HERSHEY_COMPLEX,
                1,                      // font face and scale
                Scalar(255, 255, 255),  // white
                1, LINE_AA);            // line thickness and type
      }
      imshow("Sample", image);

      std::cout << "This is the prediction" << cppflow::arg_max(output_2, 1);

      if (image.empty())
        break;
      // drawText(image);
      if (waitKey(10) >= 0)
        break;
    }
  } else {
    cout << "No capture" << endl;
    image = Mat::zeros(480, 640, CV_8UC1);
    // drawText(image);
    imshow("Sample", image);
    waitKey(0);
  }
}

void MaskadeClassifier::setup() {

}

void MaskadeClassifier::mouseDown(ci::app::MouseEvent event) {

}

void MaskadeClassifier::update() {

}

void MaskadeClassifier::draw() {
  Run();
}

}  // namespace maskade