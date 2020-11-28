#include "maskade_classifier.hpp"

#include "CinderOpenCV.hpp"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "nlohmann/json.hpp"
#include "opencv2/core/mat.hpp"

namespace maskade {

MaskadeClassifier::MaskadeClassifier() : model_(model_path_) {
}

void MaskadeClassifier::setup() {
  std::ifstream input(config_path_);
  nlohmann::json config;
  input >> config;

  font_name_ = (std::string(config["font"]));

  OpenCamera();
}

void MaskadeClassifier::update() {
  capture_ >> image_;
}

void MaskadeClassifier::draw() {
  image_.convertTo(image_, CV_32F);
  image_ /= 255.0;

  ci::app::setWindowSize(image_.cols, image_.rows);

  ci::gl::TextureRef texture = ci::gl::Texture::create(ci::fromOcv(image_));
  ci::gl::draw(texture);
  cv::cvtColor(image_, image_, cv::COLOR_BGR2RGB);

  // Assign to vector for 3 channel image
  // Souce: https://stackoverflow.com/a/56600115/2076973
  cv::Mat flat = image_.reshape(1, image_.total() * image_.channels());

  std::vector<float> img_data(image_.total() * image_.channels());
  img_data = image_.isContinuous() ? flat : flat.clone();
  cppflow::tensor tensor(img_data,
                         {1, image_.rows, image_.cols, image_.channels()});
  std::cout << tensor.dtype();
  auto dims = {224, 224};
  tensor = cppflow::resize_bilinear(tensor, cppflow::tensor(dims));

  auto output_2 = model_(tensor);

  auto argmax = cppflow::arg_max(output_2, 1).get_data<float>();

  std::string output_line = (argmax[0] == 0)
                                ? "Hey, your mask isn't on!"
                                : "Thank you for wearing your mask!";

  ci::gl::drawStringCentered(output_line,
                             glm::vec2(ci::app::getWindowWidth() / 2,
                                       ci::app::getWindowHeight() * 9 / 10),
                             ci::ColorT<float>().hex(0xffffff),
                             ci::Font(font_name_, 30));
}

void MaskadeClassifier::OpenCamera() {
  capture_.open(0);
  if (capture_.isOpened()) {
    std::cout << "Video capture is opened." << std::endl;
  } else {
    std::cout << "No video capture is available. You may need to enable "
                 "Superuser permissions."
              << std::endl;
    auto image = cv::Mat::zeros(1000, 1000, CV_8UC1);
    cv::imshow("", image);
    cv::waitKey(0);
  }
}

}  // namespace maskade