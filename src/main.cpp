#include "MJPEGWriter.h"
#include "DIP.h"

void test_cameras();
void *ypr_thread(void *threadID);
Mat side_by_side_3d_frame(Mat left, Mat right);


int
main()
{
//    test_cameras();
    pthread_t threads[1];
    int rc = pthread_create(&threads[0],NULL,ypr_thread,(void *)0);
    cout<<"Main"<<endl;
    if(rc){
        cout<<"Problem in creating thread";
        exit(-1);
    }
    MJPEGWriter test(8080);

    VideoCapture cap_left,cap_right;
    bool ok1 = cap_left.open(2);
    bool ok2 = cap_right.open(0);
//    cap.set(cv::CAP_PROP_BUFFERSIZE,1);
    if (!ok1)
    {
        printf("left cam not found ;(.\n");
    }
    if (!ok2)
    {
        printf("right cam not found ;(.\n");
    }
    Mat frame_left, frame_right, out;

    VideoCapture cap_vid;
    bool ok3 = cap_vid.open("//home//pi//Videos//KSA_history_01.mp4");
    Mat frame;
    Mat frame_vid;
    cap_left >> frame;
    resize(frame,frame,Size(1280,720));
    test.write(frame);
    frame.release();
    test.start();
    cap_vid >> frame_vid;
    while(cap_left.isOpened()&&cap_right.isOpened()){
        cap_left >> frame_left; 
        cap_right >> frame_right; 
        frame = side_by_side_3d_frame(frame_left, frame_right);
//cout<<frame.rows<<", "<<frame.cols<<endl;
        cap_vid >> frame_vid;
        frame(Rect(0,0,frame.cols/2,frame.rows)).copyTo(frame_left);
        frame(Rect(frame.cols/2,0,frame.cols/2,frame.rows)).copyTo(frame_right);
        frame = process_frame(frame_left,frame_right,frame_vid);
        test.write(frame);
        namedWindow( "image", 1 );
        imshow( "image", frame );
        waitKey(1); 
        frame.release();
        }
    test.stop();
    exit(0);

}



void error( char *msg ) {
  perror(  msg );
  exit(1);
}

int func( int a ) {
   return 2 * a;
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


void *ypr_thread(void *threadID){
    long tid = (long)threadID;
    socket_ex();
    while(1){
        cout<<"hello in thread :"<<tid<<endl;
    }
    pthread_exit(NULL);
}

 
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


void test_cameras(){
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
