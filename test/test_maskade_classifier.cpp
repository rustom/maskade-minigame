// // Read in a sample image (hopefully, this will later be from the camera feed)
//   cppflow::tensor input = cppflow::decode_jpeg(cppflow::read_file(
//       std::string("/Users/rustomichhaporia/GitHub/Cinder/my-projects/"
//                   "final-project-rustom-ichhaporia/assets/photo3.jpeg")));

//   std::cout << input;
//   // Cast the datatype of the input, expand dimensions, and change size to match
//   // the image size of the model
//   input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
//   input = input / 255.f;
//   input = cppflow::expand_dims(input, 0);
//   std::cout << input.shape();
//   auto il = {224, 224};
//   input = cppflow::resize_bilinear(input, cppflow::tensor(il));
//   std::cout << input.shape();

//   // Load in the saved model built online with Google's Teachable Machines
//   // project https://teachablemachine.withgoogle.com/train
//   cppflow::model model(
//       "/Users/rustomichhaporia/GitHub/Cinder/my-projects/"
//       "final-project-rustom-ichhaporia/assets/converted_savedmodel/"
//       "model.savedmodel");

//   // Print list of possible operations on the Tensor model
//   // std::vector<std::string> ops = model.get_operations();
//   // for (auto item : ops) {
//   //   std::cout << item << std::endl << std::endl;
//   // }

//   // Output the prediction from the model
//   auto output = model(input);
//   std::cout << output;