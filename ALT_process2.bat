::RUN YOLOV4 MODEL ON IMAGE
@ECHO OFF
python changeconfigtoWS.py
python detect.py --weights ./checkpoints/yolov4-416-ws --size 416 --model yolov4 --images ./data/images/2.jpg --crop
python changeconfigtoSB.py
echo Finish MATLAB process before continuing
pause
