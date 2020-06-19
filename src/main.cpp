#include "MJPEGWriter.h"

void *ypr_thread(void *threadID);


int
main()
{
    pthread_t threads[1];
    int rc = pthread_create(&threads[0],NULL,ypr_thread,(void *)0);
    cout<<"Main"<<endl;
    if(rc){
        cout<<"Problem in creating thread";
        exit(-1);
    }
    MJPEGWriter test(8080);

    VideoCapture cap;
    bool ok = cap.open("//home//pi//RemoteVisitorBase//mjpeg//media//SanFran.mkv");
    //bool ok = cap.open("//home//pi//RemoteVisitorBase//mjpeg//media//ZED.mkv");
    //bool ok = cap.open(0);
    cap.set(cv::CAP_PROP_BUFFERSIZE,1);
    if (!ok)
    {
        printf("no cam found ;(.\n");
        pthread_exit(NULL);
    }
    Mat frame;
    cap >> frame;
    test.write(frame);
    frame.release();
    test.start();
    while(cap.isOpened()){
        cap >> frame; 
        //resize(frame,frame,cv::Size(),0.5,0.5);
        //hconcat(frame,frame,frame);
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
  printf( "ypr = %.2ff\t%.2ff\t%.2ff\n", yaw,pitch,roll );
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
