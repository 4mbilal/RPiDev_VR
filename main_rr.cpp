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
#include <fcntl.h>      // File control definitions
#include <termios.h>    // POSIX terminal control definitions

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
void test_uart();
void USB_uart_setup();
void Mat2Buffer(unsigned char* inputBuffer, Mat inImage);
void Buffer2Mat(unsigned char* outputBuffer, Mat outImage);
void AI_detector();
void MJPEG_Streamer();
void process_frame();
void HeadTracking();
bool streamerStarted;
int socket_ex();
int getData(int sockfd);
void error(char *msg);
void cleanup();

Mat Stream_frames[4];
int Stream_frames_indx;
bool Stream_frame_status;
int obj_id;
float fps;
VideoCapture pip_vid;
int sockfd, newsockfd;
int USB_uart;
unsigned char ypr_cmd[6];


int main(){
	//Remote_Reality();
	//test_uart();
	test_AI_detector();
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
    
    thread HTthread (HeadTracking);
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
            int key = waitKey(1);
            if(key==27){
            	vid_streamer.release();
            	cleanup();
            }
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
  	case 2: sprintf(img_str,"Elephant Mountain (Al-Ula)");
  					if(!pip_vid.isOpened())
  						pip_vid.open("/home/brainiac/Videos/AlUla.mkv");
  					break;
  	case 3: sprintf(img_str,"Floating Mosque (Jeddah)");
  					if(!pip_vid.isOpened())
  						pip_vid.open("/home/brainiac/Videos/Jeddah.mkv");
  					break;
  	case 4: sprintf(img_str,"Ancient AlAhsa Script (Thaj)");
  					if(!pip_vid.isOpened())
  						pip_vid.open("/home/brainiac/Videos/NationalMuseum.mp4");
  					break;
  	case 5: sprintf(img_str,"Qasar Al-Farid (Al-Ula)");
  					if(!pip_vid.isOpened())
  						pip_vid.open("/home/brainiac/Videos/AlUla.mkv");
  					break;
  	case 6: sprintf(img_str,"Kingdom Tower (Riyadh)");
  					if(!pip_vid.isOpened())
  						pip_vid.open("/home/brainiac/Videos/Riyadh.mkv");
  					break;
  	case 7: sprintf(img_str,"%0.1f",fps);
  					break;        		
  }
  obj_id_prev = obj_id;
  int disp = 50;
  putText(Stream_frames[Stream_frames_indx],img_str,Point(20+disp,100),cv::FONT_HERSHEY_DUPLEX,1.0,CV_RGB(118, 185, 0),2);
  putText(Stream_frames[Stream_frames_indx],img_str,Point(20+640,100),cv::FONT_HERSHEY_DUPLEX,1.0,CV_RGB(118, 185, 0),2);

	Mat pip;
	if(pip_vid.isOpened()){
		pip_vid>>pip;
		if(!pip.empty()){
			//pip.copyTo(Stream_frames[Stream_frames_indx](Rect(0,0,pip.cols,pip.rows)));
			resize(pip,pip,Size(267,200));
			addWeighted(Stream_frames[Stream_frames_indx](Rect(320+disp,240,pip.cols,pip.rows)),0.5,pip,0.5,0.0,Stream_frames[Stream_frames_indx](Rect(320+disp,240,pip.cols,pip.rows)));
			addWeighted(Stream_frames[Stream_frames_indx](Rect(320+640,240,pip.cols,pip.rows)),0.5,pip,0.5,0.0,Stream_frames[Stream_frames_indx](Rect(320+640,240,pip.cols,pip.rows)));
			rectangle(Stream_frames[Stream_frames_indx], Point(320+disp,240), Point(320+disp+pip.cols,240+pip.rows), CV_RGB(64, 0, 64), 2, 8, 0);
			rectangle(Stream_frames[Stream_frames_indx], Point(320+640,240), Point(320+640+pip.cols,240+pip.rows), CV_RGB(64, 0, 64), 2, 8, 0);
		}
		else
			pip_vid.release();
  }    
  		
/*       static Mat img;
        if(img.empty()){
                img = imread("//home//pi//Pictures//KAU_logo.jpg");
        }
*/
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
    //if (!cap.open("/home/brainiac/Videos/vid0_stereo.mp4.avi")){
    if (!cap.open(0)){
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
    
//   unsigned char* ipBuffer = (unsigned char*)calloc(sizeof(unsigned char), Stream_frames[0].rows*Stream_frames[0].cols*3);
   unsigned char* ipBuffer = (unsigned char*)calloc(sizeof(unsigned char), 154587);//227x227x3

   while(cap.isOpened()){
   	//1- Capture Frame
        Stream_frames_indx = (Stream_frames_indx+1)%4;
        cap >> Stream_frames[(Stream_frames_indx+1)%4]; 
        if (Stream_frames[(Stream_frames_indx+1)%4].empty()){
        	while(1);
        }

    //2- CNN inference
        ftime(&t_start);
        Mat fr_cnn;
        //resize(Stream_frames[(Stream_frames_indx+1)%4],fr_cnn,Size(640,480));
//        Stream_frames[(Stream_frames_indx+1)%4](Rect(0,0,640,480)).copyTo(fr_cnn);
        Stream_frames[(Stream_frames_indx+1)%4](Rect(206,126,227,227)).copyTo(fr_cnn);
        imshow("AI Frame",fr_cnn);
        waitKey(1);
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
	//This function converts OpenCV style mat to Matlab style array (column-wise with separate non-interleaved channels)
    //transpose and re-order color channels in the input image
    for (int chnl=0; chnl < 3; chnl++) {
      for (int col=0; col < 227; col++) {
        for (int row=0; row < 227; row++) {
	        inputBuffer[227*227*2+col*227+row] = inImage.data[row*227*3+col*3+0];//Red
	        inputBuffer[227*227*1+col*227+row] = inImage.data[row*227*3+col*3+1];//Green
	        inputBuffer[227*227*0+col*227+row] = inImage.data[row*227*3+col*3+2];//Blue
        }//end of width
      }//end of height
    }//end of channels
    
}

void Buffer2Mat(unsigned char* outputBuffer, Mat outImage) {
    for (int j = 0; j < outImage.rows * outImage.cols; j++) {
        // BGR to RGB
        outImage.data[j * 3 + 0] = outputBuffer[j];
        outImage.data[j * 3 + 1] = outputBuffer[j];
        outImage.data[j * 3 + 2] = outputBuffer[j];
    }
}

void cleanup(){
	close(newsockfd);
	exit(0);
}

int test_AI_detector() {
		timeb t_start, t_end;
    char img_str[50];

    Mat im;
    VideoCapture cap_cam;
//    if (!cap_cam.open("/home/pi/Videos/vid2.avi"))
    if (!cap_cam.open(0))
    {
        printf("Camera could not be opened !!!\n");
        exit(-1);
    }
    //cap_cam.set(cv::CAP_PROP_FRAME_WIDTH,1280);
		//cap_cam.set(cv::CAP_PROP_FRAME_HEIGHT,480);
			    
    cap_cam >> im;
    unsigned char* ipBuffer = (unsigned char*)calloc(sizeof(unsigned char), 154587);
    //unsigned char* opBuffer = (unsigned char*)calloc(sizeof(unsigned char), im.rows*im.cols);
     
   	ftime(&t_start);
    float fps = 8.5;		
    float res = 0;
    Mat fr_cnn;
    while(1){
        cap_cam >> im;
        im(Rect(206,126,227,227)).copyTo(fr_cnn);
        Mat2Buffer(ipBuffer, fr_cnn);
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
        putText(fr_cnn,img_str,Point(20,20),cv::FONT_HERSHEY_DUPLEX,1.0,CV_RGB(118, 185, 0),2);
        imshow("Output",fr_cnn);
        waitKey(1);
    }
    free(ipBuffer);
    //free(opBuffer);
    return 0;
}

int test_MJPEG_Streamer(){
//    test_cameras();
    /*pthread_t threads[1];
    int rc = pthread_create(&threads[0],NULL,ypr_thread,(void *)0);
    cout<<"Main Thread"<<endl;
    if(rc){
        cout<<"Problem in creating thread";
        exit(-1);
    }*/
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

void HeadTracking() {
     int portno = 5000, clilen;
     char buffer[256];
     struct sockaddr_in serv_addr, cli_addr;
     int n;
     int data;
    printf("\n\tHead tracking started at port: #%d\n", portno );
    
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
     
     //Open UART
     USB_uart_setup();
  
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
//  printf( "ypr = %.2f\t\t%.2f\t\t%.2f\n", yaw,pitch,roll );
  
  yaw = -yaw + 90;
  if(yaw<0)
  	yaw = 0;
  if(yaw>180)
  	yaw = 180;
  	
  pitch = pitch + 90;
  if(pitch<0)
  	pitch = 0;
  if(pitch>180)
  	pitch = 180;

  roll = -roll + 90;
  if(roll<0)
  	roll = 0;
  if(roll>180)
  	roll = 180;
  	
  ypr_cmd[3] = (unsigned char)(yaw);
  ypr_cmd[4] = (unsigned char)(pitch);
  ypr_cmd[5] = (unsigned char)(roll);
  write( USB_uart, ypr_cmd, sizeof(ypr_cmd));
  printf( "ypr = %d\t\t%d\t\t%d\n", ypr_cmd[3],ypr_cmd[4],ypr_cmd[5] );  
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

void USB_uart_setup(){
	/* Open File Descriptor */
	//int USB = open( "/dev/ttyUSB0", O_RDWR| O_NONBLOCK | O_NDELAY );
	USB_uart = open( "/dev/ttyUSB0", O_RDWR| O_NOCTTY );
	
	/* Error Handling */
	if ( USB_uart < 0 )
	{
		cout << "Error " << errno << " opening " << "/dev/ttyUSB0" << ": " << strerror (errno) << endl;
	}
	
	/* *** Configure Port *** */
	struct termios tty;
	struct termios tty_old;
	memset (&tty, 0, sizeof tty);
	
	/* Error Handling */
	if ( tcgetattr ( USB_uart, &tty ) != 0 ) {
	   std::cout << "Error " << errno << " from tcgetattr: " << strerror(errno) << std::endl;
	}
	
	/* Save old tty parameters */
	tty_old = tty;
	
	
	
		/* Set Baud Rate */
	cfsetospeed (&tty, (speed_t)B115200);
	cfsetispeed (&tty, (speed_t)B115200);
	
	/* Setting other Port Stuff */
	tty.c_cflag     &=  ~PARENB;            // Make 8n1
	tty.c_cflag     &=  ~CSTOPB;
	tty.c_cflag     &=  ~CSIZE;
	tty.c_cflag     |=  CS8;
	
	tty.c_cflag     &=  ~CRTSCTS;           // no flow control
	tty.c_cc[VMIN]   =  1;                  // read doesn't block
	tty.c_cc[VTIME]  =  5;                  // 0.5 seconds read timeout
	tty.c_cflag     |=  CREAD | CLOCAL;     // turn on READ & ignore ctrl lines
		
		
	/* Make raw */
	cfmakeraw(&tty);
	
	/* Flush Port, then applies attributes */
	tcflush( USB_uart, TCIFLUSH );
	if ( tcsetattr ( USB_uart, TCSANOW, &tty ) != 0) {
	   std::cout << "Error " << errno << " from tcsetattr" << std::endl;
	}
		
	ypr_cmd[0] = 'Y';
	ypr_cmd[1] = 'P';
	ypr_cmd[2] = 'R';
	ypr_cmd[3] = 0;
	ypr_cmd[4] = 0;
	ypr_cmd[5] = 0;
}

void test_uart(){
	USB_uart_setup();
	int sd = 1;
	while(1){
		ypr_cmd[3] = ypr_cmd[3] + sd;
		if(ypr_cmd[3]>180){
			ypr_cmd[3] = 180;
			sd = -sd;
		}
		if(ypr_cmd[3]<1){
			ypr_cmd[3] = 1;
			sd = -sd;
		}
		int n_written = write( USB_uart, ypr_cmd, sizeof(ypr_cmd));
		cout<<endl<<n_written;
		usleep(250000);
	}
	
	/*
	unsigned char cmd[] = "INIT \r";
	int n_written = 0,
	    spot = 0;
	
	do {
	    n_written = write( USB, &cmd[spot], 1 );
	    spot += n_written;
	} while (cmd[spot-1] != '\r' && n_written > 0);
	*/
		
	//int n_written = write( USB, cmd, sizeof(cmd) -1)
	
	
	int n = 0,
	    spot = 0;
	char buf = '\0';
	
	/* Whole response*/
	char response[1024];
	memset(response, '\0', sizeof response);
	
	do {
	    n = read( USB_uart, &buf, 1 );
	    sprintf( &response[spot], "%c", buf );
	    spot += n;
	} while( buf != '\r' && n > 0);
	
	if (n < 0) {
	    std::cout << "Error reading: " << strerror(errno) << std::endl;
	}
	else if (n == 0) {
	    std::cout << "Read nothing!" << std::endl;
	}
	else {
	    std::cout << "Response: " << response << std::endl;
	}	
	
	close(USB_uart);
}