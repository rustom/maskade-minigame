# Maskade

## Description

I would like to create a program that uses Cinder to access the camera feed of my laptop and determine if the user is wearing a mask or not. Using Haar Cascades (hence Maskade), the OpenCV computer vision library can classify and track faces and should be able to differentiate those with masks from those without. Although it has nearly no documentation like much of Cinder, there is a Cinderblock for OpenCV available online which integrates with Cinder. The project seems like it would tie into current events well and be a sufficient challenge to complete, if possible. 

Currently, I am working on building an empty program that uses the OpenCV Cinderblock. I am getting an "atomic is a c11 extension" error again from the OpenCV files when trying to build. If I cannot get it to run, I can either try writing the classification code in Python with the Python flavor of OpenCV instead of the Cinderblock and only use C++ for the Cinder drawing and video feed, or I can think of a new project idea. Hopefully I will have figured this out by the end of this week. 

## Outline

Week 1: 
- [x] Building the OpenCV Cinderblock
- [x] Setting up a Cinder app that can read in the camera's video feed

Week 2: 
- [x] Using a downloaded classification model that is pretrained to determine whether or not the user is wearing a face covering
- [x] Using Cinder to draw a message explaining the results and desired action ("put on a mask!") on top of the video feed depending upon the results of the live classification

Week 3: 
- [ ] Use Cinder to detect the color of the mask
- [ ] Checking multiple people in the video and drawing arrows by the ones that don't have their masks on

Stretch goals: 
- [ ] Training my own model from scratch
- [ ] Checking whether the user has the mask on completely or just over their nose
- [ ] Checking whether the user actually has a mask on or is just covering their face with their hand 

In case I am not able to get the video stream working, I will just do the above tasks on regular image files. If I am not able to get OpenCV to work by the first week, then I will have a new project idea. 

Rustom Ichhaopria