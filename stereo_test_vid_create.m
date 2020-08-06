close all
clear all
clc

vr = VideoReader('D:\RnD\Frameworks\Matlab\ML\CNN\AI_NEOM\videos\vid0.mp4');
vw = VideoWriter('D:\RnD\Frameworks\Matlab\ML\CNN\AI_NEOM\videos\vid0_stereo');
open(vw);

% vr.CurrentTime = 2.5;
currAxes = axes;
while hasFrame(vr)
    vidFrame = readFrame(vr);
    vidFrame = imresize(vidFrame,[480,640]);
    vidFrame = cat(2,vidFrame,vidFrame);
    writeVideo(vw,vidFrame);
    image(vidFrame, 'Parent', currAxes);
    currAxes.Visible = 'off';
%     pause(1/vr.FrameRate);
end
close(vw)