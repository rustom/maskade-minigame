# Maskade Classifier

## Demo

![](./assets/demo.gif)

## Description

The Maskade Classifier is a lightweight computer vision project built with C++. It uses the user's camera to determine if they have a face mask for COVID-19 correctly placed on their face. It also features a minigame in which the user has to try to keep their face behind a floating digital mask to score points before time runs out. The name is a pun on cascade classifiers (e.g. the Haar Cascade Classifier), a common type of computer vision object detection model. 

## Features

- Plain classification: run the program and it will tell you if you're wearing a mask or not! Avoid dark lighting for best results. 
- Minigame: Press the `M` key to enter or exit the minigame mode, where you can take off your real mask and score points by keeping your face behind a floating mask for as many frames as possible within the time limit. Press the `R` key to reset the game score and timer while in progress. 
- JSON configuration file that allows you to set default parameters for the program's style and fonts. This can be found in `config/config.json`.
- The program uses a pretrained TensorFlow `savedmodel` file to classify images from your computer's camera feed. This can be found ing `assets/converted_savedmodel`. This model was created using the [Google Teachable Machines](https://teachablemachine.withgoogle.com/) project. 

## Dependencies

The program can be cloned and run locally, but it requires several libraries to be installed and linked on your computer to properly work. The dependencies are listed below: 

| Library | Purpose | Manual Linkage Required (Y/N) | Version |
| ------- | ------- | ------------------------- | ------- |
| [TensorFlow C API](https://www.tensorflow.org/install/lang_c) | Trains, caches, loads, and runs the computer vision model | Y | 2.3.1 |
| [Cppflow](https://github.com/serizba/cppflow) | Provides an interface between C++ and the TensorFlow C API to allow a Cinder app to use TensorFlow | Y | 2.0 |
| [Cinder](https://libcinder.org/) | Allows the program to draw the camera feed and text to a window | Y | 0.9.1 |
| [OpenCV for C++](https://docs.opencv.org/master/d9/df8/tutorial_root.html) | Connects C++ to your computer's camera and facilitates image data operations | Y | 4.5.0 |
| [OpenCV Cinderblock](https://github.com/cinder/Cinder-OpenCV) | Converts OpenCV image matrices to textures that Cinder can display | N | 2.4.9 |
| [JSON for C++](https://github.com/nlohmann/json) | Gives Cinder the ability to read the JSON configuration file | N | 3.9.1 |
