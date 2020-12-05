import cv2
import numpy as np
import glob


def getImagesInDir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.png'):
        image_list.append(filename)

    return image_list


def main():
    net = cv2.dnn.readNet('../../data/data2/YoLoV3/yolov3_training_final.weights', 'yolov3_testing.cfg')

    classes = []
    with open("classes.txt", "r") as f:
        classes = f.read().splitlines()

    image_paths = getImagesInDir("../../data/data2/testing/images/")

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))

    # img_array = []
    img_idx = 0
    while True:
        try:
            print("img_idx: ", img_idx)
            print("image_paths[img_idx]: ", image_paths[img_idx])

            img = cv2.imread(image_paths[img_idx])

            height, width, _ = img.shape

            blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
            net.setInput(blob)
            output_layers_names = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(output_layers_names)

            boxes = []
            confidences = []
            class_ids = []

            for output in layerOutputs:
                # print("type(output): ", type(output))
                # print("output.shape: ", output.shape)
                # print("output: ", output)
                for detection in output:
                    # print("type(detection): ", type(detection))
                    # print("detection.shape: ", detection.shape)
                    # print("detection: ", detection)
                    # print(a)
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.2:
                        center_x = int(detection[0]*width)
                        center_y = int(detection[1]*height)
                        w = int(detection[2]*width)
                        h = int(detection[3]*height)

                        x = int(center_x - w/2)
                        y = int(center_y - h/2)

                        boxes.append([x, y, w, h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i], 2))
                    color = colors[i]
                    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                    cv2.putText(img, label + " " + confidence, (x, y+20), font, 0.5, (255, 255, 255), 1)

            cv2.imwrite("../../data/data2/YoLoV3/output/images/" + image_paths[img_idx].split('/')[-1], img)
            img_idx += 1

            # img_array.append(img)

            # cv2.imshow('Image', img)
            # key = cv2.waitKey(1)
            # if key == 27:
            #     break
        except IndexError:
            break

    # print("Generating the video with the processed images.")
    # out = cv2.VideoWriter('../../data/data2/testing/videos/mask_detection_out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
    # for i in range(len(img_array)):
    #     out.write(img_array[i])
    # out.release()
    # print("Done")
    #
    # cap.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()