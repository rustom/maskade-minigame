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
