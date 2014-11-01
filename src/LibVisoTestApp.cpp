#include "cinder/app/AppNative.h"
#include "cinder/gl/gl.h"
#include "cinder/ImageIo.h"
#include "cinder/gl/Texture.h"
#include "cinder/Camera.h"

#include "CinderOpenCv.h"

#include <iostream>
#include <string>
#include <vector>
#include <stdint.h>

#include "viso.h"
#include "triangle.h"
#include "timer.h"
#include "viso_stereo.h"
#include "viso_mono.h"
#include "matrix.h"
#include "matcher.h"
#include "filter.h"
#include "reconstruction.h"

using namespace std;
using namespace ci;
using namespace ci::app;
using namespace cv;

class LibVisoTestApp : public AppNative {
  public:
	void setup();
	void update();
	void draw();
	
	void keyDown(KeyEvent event);
	
	gl::Texture	mTexture;
	CameraPersp mCameraPersp;
	
	cv::Mat K;
	cv::Mat distortion_coeff;
	double fovx;
	double fovy;
	double focalLength;
	Point2d principalPoint;
	double aspectRatio;
	
	
	cv::VideoCapture capture;
	
	VisualOdometryMono * viso;
	
	VisualOdometryMono::parameters param;
	
	Mat_<double> mTrans;
	
	Mat_<double> mPose;
	
	bool processFrame(Mat& f_);
	void initVO();
	void initScene();
	
};

/*
void PtoRandT(Mat& P_, cv::Vec3f& _R, cv::Vec3f& _T)
{
	_T = cv::Vec3f(P_.at<float>(0, 3), P_.at<float>(1, 3), P_.at<float>(2, 3));
	_R = Rodrigues(
	
}
*/

Mat_<double> Matrix2Mat(Matrix& matrix_)
{
	int columns = matrix_.m;
	int rows = matrix_.n;
	
	Mat_<double> _mat(rows, columns);
	double m_data[rows*columns];
	matrix_.getData(m_data, 0, 0, 3, 3);
	
	
	for(int i = 0; i < rows; i++)
	{
		for(int j = 0; j < columns; j++)
		{
			_mat(i, j) = (m_data[i*columns+j]);
		}
	}
	
	return _mat;
}

void LibVisoTestApp::initVO()
{
	cv::FileStorage fs;
    fs.open("/users/wolfgag/sfmSimple/out_camera_data.xml",cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        cerr << "Failed to open";
        return;
    }
    
	double imageWidth;
	double imageHeight;
	//http://blog.falklumo.com/2010/06/apple-iphone-4-camera-specs.html
	double apertureWidth = 4.54;
	double apertureHeight = 3.39;
	
    fs["Camera_Matrix"]>>K;
    fs["Distortion_Coefficients"]>>distortion_coeff;
	fs["image_Width"]>>imageWidth;
	fs["image_Height"]>>imageHeight;
	cv::Size imageSize(imageWidth, imageHeight);
	
	calibrationMatrixValues(K, imageSize, apertureWidth, apertureHeight, fovx, fovy, focalLength, principalPoint, aspectRatio);
	
	param.calib.f  = focalLength; // focal length in pixels
    param.calib.cu = principalPoint.x; // principal point (u-coordinate) in pixels
    param.calib.cv = principalPoint.y; // principal point (v-coordinate) in pixels
	
	cout << "K: " << K << ", focalLength: " << focalLength << ", principalPoint: " << principalPoint << endl;
	
	// init visual odometry
	viso = new VisualOdometryMono(param);
}

void LibVisoTestApp::initScene()
{
	mCameraPersp.setAspectRatio(aspectRatio);
	mCameraPersp.lookAt( ci::Vec3f( 0, 0, 500 ), ci::Vec3f::zero() );
	gl::setMatrices(mCameraPersp);
}

bool LibVisoTestApp::processFrame(Mat& f_)
{
	
    Mat f;
    vector<uint8_t>f_v;
    f_.copyTo(f);
    int32_t w1 = f.cols;
    int32_t h1 = f.rows;
    
    // current pose (this matrix transforms a point from the current
    // frame's camera coordinates to the first frame's camera coordinates)
    Matrix pose = Matrix::eye(4);
    
    if (f.channels() == 3)
    {
        cvtColor(f, f, CV_BGR2GRAY);
    }
    
	int k = 0;
    for(int i = 0; i < w1; i++)
    {
        for(int j = 0; j < h1; j++)
        {
			uint8_t f_jk = f.at<uint8_t>(j, k);
            f_v.push_back(f_jk);
			k++;
        }
    }

    // compute visual odometry
    int32_t dims[] = {w1,h1,w1};
    if (viso->process(f_v.data(),dims)) //data is in uint8_t
    {
        // on success, update current pose
        pose = pose * Matrix::inv(viso->getMotion());
		mPose = Matrix2Mat(pose);
		mTrans.setTo(Scalar(0));
		mTrans(3, 0) = 1;
		mTrans = mPose * mTrans;
		cout << "mTrans" << mTrans;

        // output some statistics
        double num_matches = viso->getNumberOfMatches();
        double num_inliers = viso->getNumberOfInliers();
        cout << ", Matches: " << num_matches;
        cout << ", Inliers: " << 100.0*num_inliers/num_matches << " %" << ", Current pose: " << endl;
		cout << pose << endl << endl;
    }
    else
    {
        cout << " ... failed!" << endl;
		return false;
    }

	return true;
}

void LibVisoTestApp::keyDown( KeyEvent event )
{
}

void LibVisoTestApp::setup()
{
	// The included image is copyright Trey Ratcliff
	// http://www.flickr.com/photos/stuckincustoms/4045813826/
	mTrans = Mat_<double>(4, 1);
	mTrans.setTo(Scalar(0));
	mTrans(3, 0) = 1;
	initVO();
	initScene();
	
	gl::enableDepthWrite();
	gl::disableAlphaBlending();
	
	capture.open("/Users/wolfgag/indoorPositioning/indoorPositioning/ios_test.MOV");
}

void LibVisoTestApp::update()
{
	if (!capture.isOpened()) return;
    //if (capture.grab())
    {
		cv::Mat frame1;
		capture >> frame1;
		Surface mSurface = Surface(fromOcv(frame1));
		mTexture = gl::Texture(mSurface);
		
		
		if (frame1.data && getElapsedFrames()%4 == 1)
		{
			try
			{
				processFrame(frame1);
				
			}
			catch(...)
			{
				cout << "VO error!" <<endl;
			}
		}
	}
    
}

void LibVisoTestApp::draw()
{
	gl::clear();
	
	gl::draw( mTexture );
	
	gl::pushMatrices();
		gl::rotate(ci::Vec3f(180, (getElapsedFrames()/2)% 360, 0));
		//http://stackoverflow.com/questions/9081900/reference-coordinate-system-changes-between-opencv-opengl-and-android-sensor
		gl::translate(ci::Vec3f(mTrans(0, 0)*4, 0, mTrans(0, 2)*4));
		gl::color(255, 0, 0);
		gl::drawCube( ci::Vec3f::zero(), ci::Vec3f( 2.0f, 2.0f, 2.0f ) );
		gl::color(255, 255, 255);
	gl::popMatrices();
		
}

CINDER_APP_NATIVE( LibVisoTestApp, RendererGl )
