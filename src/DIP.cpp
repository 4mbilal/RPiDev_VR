#include "DIP.h"

Mat process_frame(Mat frame_left,Mat frame_right,Mat vid_frame){
        static Mat img;
        if(img.empty()){
                img = imread("//home//pi//Pictures//KAU_logo.jpg");
        }
        Mat pip;
        int disp = 50;
        resize(vid_frame,pip,cv::Size(),0.95,0.95);
        addWeighted(frame_left(Rect(300+disp,420,pip.cols,pip.rows)),0.25,pip,0.75,0.0,frame_left(Rect(300+disp,420,pip.cols,pip.rows)));
        addWeighted(frame_right(Rect(300,420,pip.cols,pip.rows)),0.25,pip,0.75,0.0,frame_right(Rect(300,420,pip.cols,pip.rows)));
        Mat res;
        //cvtColor(frame_right,frame_right,cv::COLOR_BGR2YUV);
        putText(frame_left,"Sample Text", Point(50+disp,360), FONT_HERSHEY_COMPLEX_SMALL, 3, Scalar(200,200,250),1, 8);
        putText(frame_right,"Sample Text", Point(50,360), FONT_HERSHEY_COMPLEX_SMALL, 3, Scalar(200,200,250),1, 8);
        hconcat(frame_left,frame_right,res);
        return res;
}
