# VR_RemoteReality
Let's a person "remotely visit" a place using VR goggles. The host uses the headtracking signals to move a servo-mounted stereo camera accordingly. The CNN-SVM detector detects the custom objects in view and plays a picture-in-picture informative video.  
1- Streams stereo video to Android-based VR goggles and receives headtracking signals to actuate the servos.  
2- Detects custom objects using Squeezenet-SVM detector. See Matlab script.  
3- Tested on both Raspbery Pi and Jetson Nano. Raspberry struggles while Jetson comfortably works at >10fps.  

## *Acknowledgements*:  

1- MJPEG streamer code in its entirety belongs to JPery (github/JPery/MJPEGWriter). Minor changes in code have been made to wait for the next frame to be captured by the camera before streaming. The original code encodes/streams the old frame if newer version is not available. This leads to a false higher fps at the receiving side.    
