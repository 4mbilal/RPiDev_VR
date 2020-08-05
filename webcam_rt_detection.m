clear all
close all
clc

cam = webcam;
cam.Resolution='640x480';
% preview(cam);

tic
while(1)
    img = snapshot(cam);
%     YPred = squeezenet_Nano_predict(img);
    YPred = squeezenet_svm_Nano_predict(img);
    img_str = '';
    switch(YPred)
        	case 1 
                img_str = 'Robbie';
        	case 2
                img_str = 'Elephant Mountain';
        	case 3
                img_str = 'Jeddah Floating Mosque'
        	case 4
                img_str = 'AlAhsa Script';
        	case 5
                img_str = 'Qasar Al-Farid (Al-Ula)';
        	case 6
                img_str = 'Riyadh Kingdom Tower';
        	case 7
                img_str = '';	
    end
        
    imshow(img)
    t=toc;
    fps = 1/t;
    str = strcat(img_str,', fps =  ',num2str(fps));
    title(sprintf('%s', str))
   drawnow
   tic
end