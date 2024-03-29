#pragma once

// This header MUST be included first (do not reorder includes)
#include "opencv2/opencv.hpp"

#include "cinder/app/App.h"
#include "cinder/gl/gl.h"
#include "cppflow/cppflow.h"

namespace maskade {

/**
 * @brief Classifier class that uses the camera feed to classify whether the
 * user is wearing a mask or not and displays the results. Also contains a
 * minigame in which the user must attempt to keep a floating mask over their
 * face to score points.
 *
 */
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

  /**
   * @brief Executes the minigame actions depending on the key that is pressed
   * by the user.
   *
   * @param event the KeyEvent to parse
   */
  void keyDown(ci::app::KeyEvent event) override;

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
  int CalculatePrediction();

  /**
   * @brief Draws the appropriate message to the Cinder app based on the model's
   * prediction.
   *
   */
  void DrawPrediction(int prediction_class);

  /**
   * @brief Draws the user's score in the text box while the minigame is being
   * played.
   *
   */
  void DrawScore();

  /**
   * @brief Draws the box overlay in which text is printed on top of the camera
   * feed.
   *
   */
  void DrawTextBox();

  /**
   * @brief Draws the floating mask on the screen in various areas while the
   * user plays the minigame.
   *
   */
  void DrawMinigameMask();

  /**
   * @brief Draws the temporary win screen after a user has finished a minigame.
   *
   */
  void DrawMinigameWinScreen();

  /**
   * @brief Executes one loop of the minigame, incrementing the score if
   * necessary.
   *
   */
  void ExecuteMinigameStep(int prediction);

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
  // The Cinder color of the text box
  ci::ColorT<float> background_color_;
  // A boolean that defines whether the app is in "minigame" state
  bool in_minigame_ = false;
  // An int representing the score of the player in the minigame
  int minigame_score_ = 0;
  // The amount of time the user gets to play the minigame
  int minigame_max_time_ = 0;
  // A timer for the minigame
  ci::Timer minigame_timer_;
  // The amount of time the win screen is displayed after the minigame
  int minigame_win_screen_time_ = 4;
  // An OpenCV rectangle representing the mask
  cv::Rect rect_;
  // A cooldown for the floating mask "box"
  int mask_cooldown_ = 0;
  // The maximum value for the mask cooldown
  int max_box_cooldown_ = 20;
};

}  // namespace maskade