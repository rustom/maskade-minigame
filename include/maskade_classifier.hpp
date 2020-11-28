#pragma once

// This header MUST be included first (do not reorder includes)
#include "opencv2/opencv.hpp"

#include <string>
#include "cinder/app/App.h"
#include "cinder/gl/gl.h"
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
   * @brief Reads in a new image from the camera every time the Cinder app
   * updates.
   *
   */
  void update() override;

  /**
   * @brief Calls the appropriate functions to draw the camera feed and
   * classification on the Cinder app.
   *
   */
  void draw() override;

 private:
  /**
   * @brief Opens the laptop camera for video access using OpenCV.
   *
   */
  void OpenCamera();

  /**
   * @brief Conducts appropriate image transformations and draws the current
   * image as a texture on the Cinder app.
   *
   */
  void DrawImage();

  /**
   * @brief Passes the image into the TensorFlow model to get a prediction of
   * the user's mask status.
   *
   */
  float CalculatePrediction();

  /**
   * @brief Draws the appropriate message to the Cinder app based on the model's
   * prediction.
   *
   */
  void DrawPrediction(float prediction_class);

  // Relative path to the configuration JSON
  std::string config_path_ = "../../../../../../config/config.json";
  // Relative path to the cached model file
  std::string model_path_ =
      "../../../../../../assets/converted_savedmodel/model.savedmodel";
  // The cppflow wrapper object for a TensorFlow model
  cppflow::model model_;
  // The width of the image to be input to the model
  int model_image_width_;
  // The height of the image to be input to the model
  int model_image_height_;
  // The OpenCV object that allows Cinder to read in camera footage
  cv::VideoCapture capture_;
  // The OpenCV matrix object representing the current image being processed
  cv::Mat image_;
  // The string name of the font for printing text
  std::string font_name_;
  // The Cinder color of the font 
  ci::ColorT<float> font_color_;
  // The width of the window
  size_t window_width_;
  // The height of the window
  size_t window_height_;
};

}  // namespace maskade
