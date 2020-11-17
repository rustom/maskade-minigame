// #include "cinder/app/App.h"
// #include "cinder/app/RendererGl.h"
// #include "cinder/gl/gl.h"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/videoio.hpp>
#include "opencv2/core/mat.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <iostream>

#include "cppflow/cppflow.h"
#include "cppflow/ops.h"
#include "cppflow/model.h"
// using namespace ci;
// using namespace ci::app;
// using namespace std;
// using namespace std;
// using namespace cv;
using cv::VideoCapture;
using std::cout;
using std::endl;
using cv::Mat;
using cv::waitKey;
using cv::LINE_AA;
using cv::Scalar;
using cv::FONT_HERSHEY_COMPLEX;


void drawText(Mat & image);

int main()
{
    auto input = cppflow::fill({10, 5}, 1.0f);
    // cppflow::model model("/Users/rustomichhaporia/GitHub/Cinder/my-projects/final-project-rustom-ichhaporia/assets/converted_savedmodel/model.savedmodel");
    cppflow::model model("/Users/rustomichhaporia/GitHub/cppflow/examples/load_model/model");
    auto output = model(input);

    // std::cout << "hello";
    std::cout << output << std::endl;

    cout << "Built with OpenCV " << CV_VERSION << endl;
    cv::Mat image;
    VideoCapture capture;
    capture.open(0);
    if(capture.isOpened())
    {
        cout << "Capture is opened" << endl;
        for(;;)
        {
            capture >> image;
            if(image.empty())
                break;
            drawText(image);
            imshow("Sample", image);
            if(waitKey(10) >= 0)
                break;
        }
    }
    else
    {
        cout << "No capture" << endl;
        image = Mat::zeros(480, 640, CV_8UC1);
        drawText(image);
        imshow("Sample", image);
        waitKey(0);
    }
    return 0;
}

void drawText(Mat & image)
{
    putText(image, "Hello OpenCV",
            cv::Point(20, 50),
            FONT_HERSHEY_COMPLEX, 1, // font face and scale
            Scalar(255, 255, 255), // white
            1, LINE_AA); // line thickness and type
}


// using namespace ci;
// using namespace ci::app;
// using namespace std;

// class finalprojectApp : public App {
//   public:
// 	void setup() override;
// 	void mouseDown( MouseEvent event ) override;
// 	void update() override;
// 	void draw() override;
	
// };

// void finalprojectApp::setup()
// {
// }

// void finalprojectApp::mouseDown( MouseEvent event )
// {
// }

// void finalprojectApp::update()
// {
// }

// void finalprojectApp::draw()
// {
// 	gl::clear( Color( 0, 0, 0 ) ); 
// }

// CINDER_APP( finalprojectApp, RendererGl )
