#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <time.h>
#include <sys/timeb.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/signal.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <thread>
#include <mutex>
#include <iostream>
#include <stdlib.h>
#include <errno.h>

#include "opencv2/opencv.hpp"
#include "squeezenet_svm_predict.h"
//#include "colortest.h"
#include "MJPEGWriter.h"

using namespace std;
using namespace cv;

//Compile natively using the following command
//touch -c /home/pi/MatlabDev/MATLAB_ws/R2020a/D/RnD/Frameworks/Matlab/ML/CNN/AI_NEOM/RemoteReality/codegen/exe/squeezenet_svm_predict/*.*;make  -f squeezenet_svm_predict_rtw.mk all MATLAB_WORKSPACE="/home/pi/MatlabDev/MATLAB_ws/R2020a" -C /home/pi/MatlabDev/MATLAB_ws/R2020a/D/RnD/Frameworks/Matlab/ML/CNN/AI_NEOM/RemoteReality/codegen/exe/squeezenet_svm_predict


//function definitions in this file
void Remote_Reality();
int test_AI_detector();
int test_MJPEG_Streamer();
void test_single_camera();
void Mat2Buffer(unsigned char* inputBuffer, Mat inImage);
void Buffer2Mat(unsigned char* outputBuffer, Mat outImage);
void AI_detector();
void MJPEG_Streamer();
void process_frame();
bool streamerStarted;
void *ypr_thread(void *threadID);
int socket_ex();
int getData(int sockfd);
void error(char *msg);

Mat Stream_frames[4];
int Stream_frames_indx;
bool Stream_frame_status;
int obj_id;
float fps;
VideoCapture pip_vid;


int main(){
	Remote_Reality();
	//test_AI_detector();
	//test_MJPEG_Streamer();
	//test_single_camera();
}

void Remote_Reality(){
	timeb t_start, t_end;
	ftime(&t_start);
	fps = 5;
	float msec;
	char img_str[50];
	obj_id = 7;
    cout<<endl<<"Starting Remote Reality Application ....";
    Stream_frames_indx = -1;
    Stream_frame_status = false;
    thread AIthread (AI_detector);
    while(!Stream_frame_status){    //wait till all frames are filled up
        cout<<".";
        usleep(500000);
    }
    	
    printf("\nMain Thread");
   
   MJPEGWriter vid_streamer(8080);
   streamerStarted = false;

   while(1){
   			do{
   				ftime(&t_end);
					msec = (float)((t_end.time - t_start.time) * 1000 + (t_end.millitm - t_start.millitm));
				}while(msec<100);//Don't go faster than 10 fps
        fps = 0.99*fps + 0.01*(1000/msec);
        ftime(&t_start);
        //sprintf(img_str,", %0.1f",fps);
        //putText(Stream_frames[Stream_frames_indx],img_str,Point(20,20),cv::FONT_HERSHEY_DUPLEX,1.0,CV_RGB(118, 185, 0),2);
        if(streamerStarted){
        		process_frame();
            vid_streamer.write(Stream_frames[Stream_frames_indx]);
            namedWindow( "Stream Frame", 1 );
            imshow( "Stream Frame", Stream_frames[Stream_frames_indx] );
            waitKey(1);
            //Stream_frames[Stream_frames_indx].release();
        }
        else{
            vid_streamer.write(Stream_frames[Stream_frames_indx]);
            //Stream_frames[Stream_frames_indx].release();
            vid_streamer.start();
            streamerStarted = true;
        }
   }
}
 
void process_frame(){
	char img_str[50];
	static int obj_id_prev;
	if(obj_id_prev!=obj_id){
		if(pip_vid.isOpened())
			pip_vid.release();
	}
		
	switch(obj_id){
  	case 1: sprintf(img_str,"Robbie %0.1f",fps);
  					break;
  	case 2: sprintf(img_str,"Elephant Mountain %0.1f",fps);
  					if(!pip_vid.isOpened())
  						pip_vid.open("/home/brainiac/Videos/AlUla.mkv");
  					break;
  	case 3: sprintf(img_str,"Jeddah Floating Mosque %0.1f",fps);
  					if(!pip_vid.isOpened())
  						pip_vid.open("/home/brainiac/Videos/Jeddah.mkv");
  					break;
  	case 4: sprintf(img_str,"AlAhsa Script %0.1f",fps);
  					if(!pip_vid.isOpened())
  						pip_vid.open("/home/brainiac/Videos/NationalMuseum.mp4");
  					break;
  	case 5: sprintf(img_str,"Qasar Al-Farid (Al-Ula) %0.1f",fps);
  					if(!pip_vid.isOpened())
  						pip_vid.open("/home/brainiac/Videos/AlUla.mkv");
  					break;
  	case 6: sprintf(img_str,"Riyadh Kingdom Tower %0.1f",fps);
  					if(!pip_vid.isOpened())
  						pip_vid.open("/home/brainiac/Videos/Riyadh.mkv");
  					break;
  	case 7: cout<<"";
  					break;        		
  }
  obj_id_prev = obj_id;
  putText(Stream_frames[Stream_frames_indx],img_str,Point(20,20),cv::FONT_HERSHEY_DUPLEX,1.0,CV_RGB(118, 185, 0),2);
  	Mat pip;
  	if(pip_vid.isOpened()){
  		pip_vid>>pip;
  		pip.copyTo(Stream_frames[Stream_frames_indx](Rect(0,0,pip.cols,pip.rows)));
    }    
  		
 /*       static Mat img;
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
        return res;*/
}

void AI_detector(){
	timeb t_start, t_end;
	ftime(&t_start);
	float fps,msec;
	int res = 7;
  int scores[7];
  for(int i=0;i<7;i++)
      scores[i] = 0;
        	
    printf("\n\tAI  Detector started");
    VideoCapture cap;
    if (!cap.open("/home/brainiac/Videos/vid0_stereo.mp4.avi")){
    //if (!cap.open(0)){
        printf("\n\tCamera could not be opened");
        return;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,1280);
		cap.set(cv::CAP_PROP_FRAME_HEIGHT,480);
   
   for(int x=0;x<10;x++){ //let the camera stabilize
	   cap >> Stream_frames[0]; 
	   cap >> Stream_frames[1]; 
	   cap >> Stream_frames[2]; 
	   cap >> Stream_frames[3]; 
 }
   Stream_frames_indx = 0;
    
   unsigned char* ipBuffer = (unsigned char*)calloc(sizeof(unsigned char), Stream_frames[0].rows*Stream_frames[0].cols*3);

   while(cap.isOpened()){
   	//1- Capture Frame
        Stream_frames_indx = (Stream_frames_indx+1)%4;
        cap >> Stream_frames[(Stream_frames_indx+1)%4]; 
    //2- CNN inference
        ftime(&t_start);
        Mat fr_cnn;
        //resize(Stream_frames[(Stream_frames_indx+1)%4],fr_cnn,Size(640,480));
        Stream_frames[(Stream_frames_indx+1)%4](Rect(0,0,640,480)).copyTo(fr_cnn);
        //imshow("AI Frame",fr_cnn);
        //waitKey(1);
        Mat2Buffer(ipBuffer, fr_cnn);
        res = squeezenet_svm_predict(ipBuffer);
        Stream_frame_status = true;
    //3- Track score    
				scores[res-1] = scores[res-1] + 2;
        if(scores[res-1]>20)
            scores[res-1] = 20;
        for(int i=0;i<7;i++){
            scores[i] = scores[i] - 1; 
            if(scores[i]<0)
                scores[i] = 0;
        }
        int max_score = 5;
        res = 7;
        for(int i=0;i<7;i++){
            if(scores[i]>max_score){
                max_score = scores[i];
                res = i + 1;
            }
        }
		//4- Report score and timing
				obj_id = res;                
   			/*ftime(&t_end);
				msec = (float)((t_end.time - t_start.time) * 1000 + (t_end.millitm - t_start.millitm));
				fps = 0.99*fps + 0.01*(1000/msec);
        cout<<res<<", "<<fps<<endl;*/
    }
}

void Mat2Buffer(unsigned char* inputBuffer, Mat inImage) {
    for (int j = 0; j < inImage.rows * inImage.cols; j++) {
        // BGR to RGB
        inputBuffer[2 * inImage.rows * inImage.cols + j] = (inImage.data[j * 3 + 0]);
        inputBuffer[1 * inImage.rows * inImage.cols + j] = (inImage.data[j * 3 + 1]);
        inputBuffer[0 * inImage.rows * inImage.cols + j] = (inImage.data[j * 3 + 2]);
    }
}

void Buffer2Mat(unsigned char* outputBuffer, Mat outImage) {
    for (int j = 0; j < outImage.rows * outImage.cols; j++) {
        // BGR to RGB
        outImage.data[j * 3 + 0] = outputBuffer[j];
        outImage.data[j * 3 + 1] = outputBuffer[j];
        outImage.data[j * 3 + 2] = outputBuffer[j];
    }
}

int test_AI_detector() {
		timeb t_start, t_end;
    char img_str[50];

    Mat im;
    VideoCapture cap_cam;
    if (!cap_cam.open(0))
    {
        printf("Camera could not be opened !!!\n");
        exit(-1);
    }
    cap_cam >> im;
    unsigned char* ipBuffer = (unsigned char*)calloc(sizeof(unsigned char), im.rows*im.cols*3);
    unsigned char* opBuffer = (unsigned char*)calloc(sizeof(unsigned char), im.rows*im.cols);
     
   	ftime(&t_start);
    float fps = 8.5;		
    float res = 0;
    while(1){
        cap_cam >> im;
        Mat2Buffer(ipBuffer, im);
        res = squeezenet_svm_predict(ipBuffer);
        /*colortest(ipBuffer,opBuffer);
        Buffer2Mat(opBuffer,im);*/
        ftime(&t_end);
				float msec = (float)((t_end.time - t_start.time) * 1000 + (t_end.millitm - t_start.millitm));
        fps = 0.999*fps + 0.001*(1000/msec);
        ftime(&t_start);
        switch(int(res)){
        	case 1: sprintf(img_str,"Robbie, %0.1f",fps);
        					break;
        	case 2: sprintf(img_str,"Elephant Mountain, %0.1f",fps);
        					break;
        	case 3: sprintf(img_str,"Jeddah Floating Mosque, %0.1f",fps);
        					break;
        	case 4: sprintf(img_str,"AlAhsa Script, %f",fps);
        					break;
        	case 5: sprintf(img_str,"Qasar Al-Farid (Al-Ula), %0.1f",fps);
        					break;
        	case 6: sprintf(img_str,"Riyadh Kingdom Tower, %0.1f",fps);
        					break;
        	case 7: sprintf(img_str,", %0.1f",fps);
        					break;        		
        }
        putText(im,img_str,Point(20,20),cv::FONT_HERSHEY_DUPLEX,1.0,CV_RGB(118, 185, 0),2);
        imshow("Output",im);
        waitKey(1);
    }
    free(ipBuffer);
    free(opBuffer);
    return 0;
}

int test_MJPEG_Streamer(){
//    test_cameras();
    pthread_t threads[1];
    int rc = pthread_create(&threads[0],NULL,ypr_thread,(void *)0);
    cout<<"Main Thread"<<endl;
    if(rc){
        cout<<"Problem in creating thread";
        exit(-1);
    }
    MJPEGWriter test(8080);

    VideoCapture cap;
    if (!cap.open(0))
    {
        printf("cam not found ;(.\n");
    }
    
    Mat frame;
    cap >> frame;
    resize(frame,frame,Size(1280,720));
    test.write(frame);
    frame.release();
    test.start();
    cap >> frame;
    while(cap.isOpened()){
        cap >> frame; 
        resize(frame,frame,Size(1280,720));
        test.write(frame);
        namedWindow( "image", 1 );
        imshow( "image", frame );
        waitKey(1); 
        frame.release();
        }
    test.stop();
    exit(0);
}

void *ypr_thread(void *threadID){
    long tid = (long)threadID;
    cout<<"Thread started";
    socket_ex();
    while(1){
        cout<<"hello in thread :"<<tid<<endl;
    }
    pthread_exit(NULL);
}

int socket_ex() {
     int sockfd, newsockfd, portno = 5000, clilen;
     char buffer[256];
     struct sockaddr_in serv_addr, cli_addr;
     int n;
     int data;

     printf( "using port #%d\n", portno );
    
     sockfd = socket(AF_INET, SOCK_STREAM, 0);
     if (sockfd < 0) 
         error( const_cast<char *>("ERROR opening socket") );
     bzero((char *) &serv_addr, sizeof(serv_addr));

     serv_addr.sin_family = AF_INET;
     serv_addr.sin_addr.s_addr = INADDR_ANY;
     serv_addr.sin_port = htons( portno );
     if (bind(sockfd, (struct sockaddr *) &serv_addr,
              sizeof(serv_addr)) < 0) 
       error( const_cast<char *>( "ERROR on binding" ) );
     listen(sockfd,5);
     clilen = sizeof(cli_addr);
  
     //--- infinite wait on a connection ---
     while ( 1 ) {
        printf( "waiting for new client...\n" );
        if ( ( newsockfd = accept( sockfd, (struct sockaddr *) &cli_addr, (socklen_t*) &clilen) ) < 0 )
            error( const_cast<char *>("ERROR on accept") );
        printf( "opened new communication with client\n" );
        while ( 1 ) {
	     //---- wait for a number from client ---
             data = getData( newsockfd );
             //printf( "got %d\n", data );
             if ( data < 0 ) 
                break;
                
           //  data = func( data );

             //--- send new data back --- 
	    // printf( "sending back %d\n", data );
            // sendData( newsockfd, data );
	}
        close( newsockfd );

        //--- if -2 sent by client, we can quit ---
        if ( data == -2 )
          break;
     }
     return 0; 
}


void sendData( int sockfd, int x ) {
  int n;

  char buffer[32];
  sprintf( buffer, "%d\n", x );
  if ( (n = write( sockfd, buffer, strlen(buffer) ) ) < 0 )
    error( const_cast<char *>( "ERROR writing to socket") );
  buffer[n] = '\0';
}

int getData( int sockfd ) {
 char buffer[12];
  float yaw,pitch,roll;
  int n;

  if ( (n = read(sockfd,buffer,12) ) < 0 )
    error( const_cast<char *>( "ERROR reading from socket") );
  char b1[] = {buffer[3],buffer[2],buffer[1],buffer[0]};
  char b2[] = {buffer[7],buffer[6],buffer[5],buffer[4]};
  char b3[] = {buffer[11],buffer[10],buffer[9],buffer[8]};
  memcpy(&yaw,&b1,sizeof(yaw));
  memcpy(&pitch,&b2,sizeof(pitch));
  memcpy(&roll,&b3,sizeof(roll));
  printf( "ypr = %.2f\t\t%.2f\t\t%.2f\n", yaw,pitch,roll );
  return 0;
}

void error( char *msg ) {
  perror(  msg );
  exit(1);
}


void test_single_camera(){
    Mat frame;
    VideoCapture cap;
    bool ok = cap.open(0);
    if (!ok)
    {
        printf("Camera could not be opened....");
        exit(0);
    }

    namedWindow( "Live Video", 1 );
   
    while(cap.isOpened()){
        cap >> frame; 
        imshow( "Live Video", frame );
        waitKey(1); 
    }
}

/*




 
Mat side_by_side_3d_frame(Mat left, Mat right){
    Mat out;
    float left_scale_correction = 1.15*1.5;
    resize(left,left,Size(0,0),left_scale_correction,left_scale_correction);
    resize(right,right,Size(0,0),1.5,1.5);
    
    int r_left = (left.rows/2)-360;
    int c_left = (left.cols/2)-320;
    int r_right = (right.rows/2)-360;
    int c_right = (right.cols/2)-320;

    int c_left_offset = 180;
    int r_left_offset = 0;
    int c_right_offset = -160;
    int r_right_offset = 0;
    int win_width = 640;
    int win_height = 720;

hconcat(left(Rect(c_left+c_left_offset,r_left+r_left_offset,win_width,win_height)),right(Rect(c_right+c_right_offset,r_right+r_right_offset,win_width,win_height)),out);
//    hconcat(left,right,out);
//    cout<<out.rows<<", "<<out.cols<<endl;
	return out;
}


void test_dual_cameras(){
    Mat frame_left,frame_right;
    VideoCapture cap1,cap2;
    bool ok1 = cap1.open(0);
    bool ok2 = cap2.open(2);
    if (!ok1)
    {
        printf("Cam not found ;(.\n");
        exit(0);
    }
    if (!ok2)
    {
        printf("Cam not found ;(.\n");
        exit(0);
    }
    
    Mat frame;
    namedWindow( "Live Video", 1 );
    cap1 >> frame_left; 
    cap2 >> frame_right; 
    
    while(cap1.isOpened()&&cap2.isOpened()){
        cap1 >> frame_left; 
        cap2 >> frame_right; 
        hconcat(frame_left,frame_right,frame);
        imshow( "Live Video", frame );
        waitKey(1); 
        frame.release();
        }
}


*/
