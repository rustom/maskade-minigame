#pragma once

// This header MUST be included first (do not reorder includes)
#include "opencv2/opencv.hpp"
#include <string>
#include "cinder/app/App.h"
#include "cppflow/cppflow.h"


namespace maskade {

class MaskadeClassifier : public ci::app::App {
 public:
  /**
   * @brief Default constructor that initializes the model using the model path. 
   *
   */
  MaskadeClassifier();

  /**
   * @brief Sets up configuration file variables and opens camera for Cinder. 
   * 
   */
  void setup() override;

  /**
   * @brief Reads in a new image from the camera every time the Cinder app updates. 
   * 
   */
  void update() override;

  /**
   * @brief Calls the appropriate functions to draw the camera feed and classification on the Cinder app. 
   * 
   */
  void draw() override;

 private:
  /**
   * @brief Opens the laptop camera for video access. 
   * 
   */
  void OpenCamera();

  std::string config_path_ = "../../../../../../config/config.json";
  std::string model_path_ =
      "../../../../../../assets/converted_savedmodel/model.savedmodel";
  cppflow::model model_;
  cv::VideoCapture capture_;
  cv::Mat image_;
  std::string font_name_;
};

}  // namespace maskade
