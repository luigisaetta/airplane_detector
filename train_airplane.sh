# to train on airplane dataset
MODEL=yolov8x.yaml
EPOCHS=160

echo
echo "Training airplane detection for" $EPOCHS "epochs using" $MODEL "..."
echo
date
echo
yolo detect train data=/home/datascience/AirplaneDetector-8/data.yaml model=$MODEL epochs=$EPOCHS imgsz=640

echo
date

