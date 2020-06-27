 
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/signal.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>

using namespace cv;
using namespace std;

Mat process_frame(Mat frame_left,Mat frame_right,Mat vid_frame);
