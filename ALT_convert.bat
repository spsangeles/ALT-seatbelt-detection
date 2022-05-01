::CONVERT DARKNET WEIGHTS TO TENSORFLOW
python changeconfigtoWS.py
python save_model.py --weights ./data/windshield.weights --output ./checkpoints/yolov4-416-ws --input_size 416 --model yolov4 

python changeconfigtoSB.py
python save_model.py --weights ./data/seatbelt.weights --output ./checkpoints/yolov4-416-sb --input_size 416 --model yolov4 

