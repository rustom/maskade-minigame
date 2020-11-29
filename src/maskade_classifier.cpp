#include "maskade_classifier.hpp"

#include "CinderOpenCV.hpp"
#include "cinder/app/RendererGl.h"
#include "nlohmann/json.hpp"
#include "opencv2/core/mat.hpp"

namespace maskade {

MaskadeClassifier::MaskadeClassifier() : model_(model_path_) {
}

void MaskadeClassifier::setup() {
  // Reads config variables from JSON file
  std::ifstream input(config_path_);
  nlohmann::json config;
  input >> config;

  // Set font variables
  font_name_ = std::string(config["font"]);
  font_color_ = ci::ColorT<float>().hex(
      uint32_t(std::stoull(std::string(config["font_color"]), 0, 16)));

  // Set dimensions for model input
  model_image_width_ = config["model_image_width"].get<int>();
  model_image_height_ = config["model_image_height"].get<int>();

  // Begin collecting video
  OpenCamera();
}

void MaskadeClassifier::update() {
  // Read in the most recent snapshot from video feed to current image
  capture_ >> image_;
  // Converts image data to OpenCV data type
  image_.convertTo(image_, CV_32F);
  // Normalizes the RGB values of the data in the image
  image_ /= 255.0;
}

void MaskadeClassifier::draw() {
  // Executes the drawing and calculation heartbeat functions

  DrawImage();

  int prediction = CalculatePrediction();

  DrawPrediction(prediction);
}

void MaskadeClassifier::keyDown(ci::app::KeyEvent event) {
  switch (event.getCode()) {
    case ci::app::KeyEvent::KEY_m: {
      in_minigame_ = (in_minigame_) ? false : true;
      break;
    }
    case ci::app::KeyEvent::KEY_r: {
      // Reset the minigame
      break;
    }
    case ci::app::KeyEvent::KEY_p: {

      break;
    }
  }
}

void MaskadeClassifier::OpenCamera() {
  // Opens the default video feed accessible by OpenCV if possible
  capture_.open(0);
  if (capture_.isOpened()) {
    std::cout << "Video capture is opened." << std::endl;

    // Read in first image to set window size
    update();

    // Set window size to match the size of the video feed
    ci::app::setWindowSize(image_.cols, image_.rows);
  }

  else {
    // Prints error message if camera cannot be opened
    std::cout << "No video capture is available. You may need to enable "
                 "Superuser permissions."
              << std::endl;
    // Display empty image
    auto image = cv::Mat::zeros(1000, 1000, CV_8UC1);
    cv::imshow("", image);
    cv::waitKey(0);
  }
}

void MaskadeClassifier::DrawImage() {
  // Create separate Mat that is resized for drawing to the app's full dimensions
  cv::Mat full_window_image;

  cv::resize(image_, full_window_image, cv::Size(ci::app::getWindowWidth(), ci::app::getWindowHeight()));

  // Creates Cinder texture from OpenCV Mat
  ci::gl::TextureRef texture = ci::gl::Texture::create(ci::fromOcv(full_window_image));
  // Draw texture on the Cinder app
  ci::gl::draw(texture);
}

float MaskadeClassifier::CalculatePrediction() {
  // Switches the color schema of the image form BGR (openCV native format) to
  // RGB (TensorFlow native format)
  cv::cvtColor(image_, image_, cv::COLOR_BGR2RGB);

  // Assign to vector for 3 channel image
  // Souce: https://stackoverflow.com/a/56600115/2076973
  cv::Mat flat_mat = image_.reshape(1, image_.total() * image_.channels());

  // Create a vector of floats representing the colors of each pixel (flattened
  // image matrix)
  std::vector<float> flat_data(image_.total() * image_.channels());
  flat_data = image_.isContinuous() ? flat_mat : flat_mat.clone();

  // Create a TensorFlow tensor with the appropriate shape
  cppflow::tensor input_tensor(
      flat_data, {1, image_.rows, image_.cols, image_.channels()});

  // Resize input tensor to fit the dimensions of image that the model is
  // expecting (change from video dimensions to model dimensions)
  input_tensor = cppflow::resize_bilinear(
      input_tensor, cppflow::tensor({model_image_width_, model_image_height_}));

  // Get output probabilities from the model
  auto output = model_(input_tensor);

  // Select the class with the highest likelihood
  auto argmax = cppflow::arg_max(output, 1).get_data<int>();

  return argmax[0];
}

void MaskadeClassifier::DrawPrediction(int prediction_class) {
  // Draw background box so text can more easily be seen
  ci::Rectf text_box(glm::vec2(ci::app::getWindowWidth() * 1 / 10, ci::app::getWindowHeight() * 8 / 10),
                     glm::vec2(ci::app::getWindowWidth() * 9 / 10, ci::app::getWindowHeight()));

  // Dark grey box
  ci::gl::color(text_box_color_);
  ci::gl::drawSolidRoundedRect(text_box, 15);

  // Reset brush for drawing images
  ci::gl::color(ci::ColorT<float>().hex(0xffffff));

  // Determine message based on calculated classification
  std::string prediction_line = (prediction_class == 0)
                                ? "Hey, your mask isn't on!"
                                : "Thank you for wearing your mask!";

  ci::gl::drawStringCentered(
      prediction_line, glm::vec2(ci::app::getWindowWidth() / 2, ci::app::getWindowHeight() * 8.5 / 10),
      font_color_, ci::Font(font_name_, ci::app::getWindowHeight() / 20));

  std::string score_line = "Try to score points by keeping the mask on your face! Your score is: " + std::to_string(minigame_score_);

  ci::gl::drawStringCentered(
      score_line, glm::vec2(ci::app::getWindowWidth() / 2, ci::app::getWindowHeight() * 9.5 / 10),
      font_color_, ci::Font(font_name_, ci::app::getWindowHeight() / 35));  
}

}  // namespace maskade