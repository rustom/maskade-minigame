#include "maskade_classifier.hpp"

#include "CinderOpenCV.hpp"
#include "cinder/app/RendererGl.h"
#include "nlohmann/json.hpp"
#include "opencv2/core/mat.hpp"
#include <random>

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

  minigame_max_time_ = config["minigame_max_time"].get<int>();

  // Begin collecting video
  OpenCamera();

  rect_ = cv::Rect(0, 0, 0, 0);
}

void MaskadeClassifier::update() {
  // Read in the most recent snapshot from video feed to current image
  capture_ >> image_;
  // Converts image data to OpenCV data type
  image_.convertTo(image_, CV_32F);
  // Normalizes the RGB values of the data in the image
  image_ /= 255.0;
  // Flip the image across the y axis
  cv::flip(image_, image_, 1);
}

void MaskadeClassifier::draw() { 
  if (in_minigame_) {
    DrawMinigameBox();
  }

  // Executes the drawing and calculation heartbeat functions
  DrawImage();

  DrawTextBox();

 
  int prediction = CalculatePrediction();

  if (in_minigame_) {
    ExecuteMinigameStep(prediction);
    DrawScore(); 

    if (minigame_timer_.getSeconds() >= minigame_max_time_) {
      DrawMinigameWinScreen();
    }


    if (minigame_timer_.getSeconds() >= minigame_max_time_ + minigame_win_screen_time_) {
      minigame_timer_.start(0);
      minigame_score_ = 0;
    }
  }

  else {
    DrawPrediction(prediction); 
  }
}

void MaskadeClassifier::keyDown(ci::app::KeyEvent event) {
  switch (event.getCode()) {
    case ci::app::KeyEvent::KEY_m: {
      in_minigame_ = (in_minigame_) ? false : true;
      // minigame_timer_ = (in_minigame_) ? minigame_max_time_ : 0.0;
      if (in_minigame_) {
        minigame_timer_.start(0);
      } 
      minigame_score_ = 0;
      break;
    }
    case ci::app::KeyEvent::KEY_r: {
      // Resets the minigame score and time
      minigame_timer_.start(0);
      minigame_score_ = 0;
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
  // Create separate Mat that is resized for drawing to the app's full
  // dimensions
  cv::Mat full_window_image;

  cv::resize(image_, full_window_image,
             cv::Size(ci::app::getWindowWidth(), ci::app::getWindowHeight()));

  // Creates Cinder texture from OpenCV Mat
  ci::gl::TextureRef texture =
      ci::gl::Texture::create(ci::fromOcv(full_window_image));
  // Draw texture on the Cinder app
  ci::gl::draw(texture);
}

int MaskadeClassifier::CalculatePrediction() {
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

  // for (auto item : model_.get_operations()) {
  //   std::cout << item << std::endl;
  // }

  // Get output probabilities from the model
  auto output = model_(input_tensor);

  // Select the class with the highest likelihood
  auto argmax = cppflow::arg_max(output, 1).get_data<int>();

  return argmax[0];
}

void MaskadeClassifier::DrawPrediction(int prediction_class) {
  // Determine message based on calculated classification
  std::string prediction_line = (prediction_class == 0)
                                    ? "Hey, your mask isn't on!"
                                    : "Thank you for wearing your mask!";

  ci::gl::drawStringCentered(
      prediction_line,
      glm::vec2(ci::app::getWindowWidth() / 2,
                ci::app::getWindowHeight() * 9 / 10),
      font_color_, ci::Font(font_name_, ci::app::getWindowHeight() / 20));
}

void MaskadeClassifier::DrawScore() {
      std::string score_line =
        "Score points by keeping your mask on! Your score is: " +
        std::to_string(minigame_score_);

    ci::gl::drawStringCentered(
        score_line,
        glm::vec2(ci::app::getWindowWidth() / 2,
                  ci::app::getWindowHeight() * 8.5 / 10),
        font_color_, ci::Font(font_name_, ci::app::getWindowHeight() / 25));

    std::string time_line = "Seconds left: " + std::to_string(minigame_max_time_ - (int) minigame_timer_.getSeconds());

    ci::gl::drawStringCentered(
        time_line,
        glm::vec2(ci::app::getWindowWidth() / 2,
                  ci::app::getWindowHeight() * 9.5 / 10),
        font_color_, ci::Font(font_name_, ci::app::getWindowHeight() / 25));
}

void MaskadeClassifier::DrawTextBox() {
  // Draw background box so text can more easily be seen
  ci::Rectf text_box(glm::vec2(ci::app::getWindowWidth() * 1 / 10,
                               ci::app::getWindowHeight() * 8 / 10),
                     glm::vec2(ci::app::getWindowWidth() * 9 / 10,
                               ci::app::getWindowHeight()));

  // Dark grey box
  ci::gl::color(text_box_color_);
  ci::gl::drawSolidRoundedRect(text_box, 15);

  // Reset brush for drawing images
  ci::gl::color(ci::ColorT<float>().hex(0xffffff));

}

void MaskadeClassifier::DrawMinigameBox() {
  if (box_cooldown_ > 0) {
    --box_cooldown_;
    cv::rectangle(image_, rect_, cv::Scalar(255, 255, 0), 50, 8, 0);
    return;
  }
  int margin = 100;
  int box_height = 50;
  int box_width = 100;
  // Used to obtain a seed for the random number engine
  std::random_device rd;   
  // Gets random position from distribution
  std::mt19937 gen(rd());  
  // Distribution of possible x values
  std::uniform_int_distribution<> width_dist(0, image_.cols - margin);
  std::uniform_int_distribution<> height_dist(0, image_.rows- margin);

  int xpos = width_dist(gen);
  int ypos = height_dist(gen);

  rect_ = cv::Rect(xpos, ypos, box_width, box_height);

  box_cooldown_ = max_box_cooldown_;
}

void MaskadeClassifier::DrawMinigameWinScreen() {
  ci::Rectf background_color(glm::vec2(0, ci::app::getWindowHeight() / 4), glm::vec2(ci::app::getWindowWidth(), ci::app::getWindowHeight() * 3 / 4));

  // Dark grey box
  ci::gl::color(text_box_color_);
  ci::gl::drawSolidRect(background_color);

  // Reset brush for drawing images
  ci::gl::color(ci::ColorT<float>().hex(0xffffff));

  std::string win_message = "Game over! Your score: " + std::to_string(minigame_score_);

    ci::gl::drawStringCentered(
        win_message,
        ci::app::getWindowCenter(),
        font_color_, ci::Font(font_name_, ci::app::getWindowHeight() / 10));
}

void MaskadeClassifier::ExecuteMinigameStep(int prediction) {
  if (prediction != 0) {
    ++minigame_score_;
  }
}

}  // namespace maskade