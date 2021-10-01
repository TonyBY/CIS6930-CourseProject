# # Testing YoLo
# ./darknet detector test ../../data/data2/obj.data cfg/yolov3_training.cfg ../../data/data2/YoLoV3/yolov3_training_final.weights -dont_show -ext_output ../../data/data2/test.txt ../../data/data2/YoLoV3/result.txt

# Translate YoLo annotation from VOC annotation.
python convert_voc_to_yolo.py

#Copy Yolo annotation to the images folder
scp /home/tony/CIS6930-CourseProject/part3/data/data2/training/annotations_yolo/*.txt /home/tony/CIS6930-CourseProject/part3/data/data2/training/images 

#Check GPU
nvidia-smi

# Train YoLo
CUDA_VISIBLE_DEVICES=2,3 ./darknet detector train ../../data/data2/obj.data cfg/yolov3_training.cfg ../../data/data2/YoLoV3/darknet53.conv.74 ../../data/data2/YoLoV3/darknet53.conv.74 -dont_show


# Test YolO: Generate testing output to the data2/testing/images/*.txt
python darknet_images.py --input ../../data/data2/testing.txt --batch_size 4 --weights ../../data/data2/YoLoV3/yolov3_training_final.weights --dont_show --save_labels --config_file cfg/yolov3_training.cfg --data_file ../../data/data2/obj.data


# Train FasterRCNN
python MaskDetection_FasterRCNN_main.py -mode train

# Test FasterRCNN
python MaskDetection_FasterRCNN_main.py -mode eval

#calculate mAP
python main.py