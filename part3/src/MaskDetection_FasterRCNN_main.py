from torchvision import transforms
import os
import argparse
import glob
import time

from MaskDetection_FasterRCNN_utils import *
from MaskDetection_FasterRCNN_MaskDataset import MaskDataset

class_list = ['without_mask', 'with_mask', 'mask_weared_incorrect']
BATCH_SIZE = 4
LEARNING_RATE = 0.001
BEST_LOSS = 0.001


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='A CNN based image colorizer.'
    )
    parser.add_argument('-data_path', default='../data/data2/',
                        help='data path of file containing the data')
    # parser.add_argument('-labels_path', default='../data/data2/annotations/',
    #                     help='data path of file containing the annotations')
    parser.add_argument('-model_path', default='/home/tony/CIS6930-CourseProject/part3/data/data2/FasterRCNN/checkpoints/model-epoch-253-losses-0.0000.pth',
                        help='Pre-trained model path of FasterRCNN')
    parser.add_argument('-mode', default='eval', help='Option: train/eval')

    return parser.parse_args(args)

def save_prediction(prediction, tmp_file):
    boxes = prediction['boxes']
    labels = prediction['labels']
    score = prediction['scores']

    with open(tmp_file, "w") as new_f:
        for i in range(len(boxes)):
            ## split a line by spaces.
            ## "c" stands for center and "n" stands for normalized
            obj_name = class_list[labels[i]]
            box = boxes[i]
            left = float(box[0])
            bottom = float(box[1])
            right = float(box[2])
            top = float(box[3])
            # x_c_n = float(x_min + x_max)/2
            # y_c_n = float(y_min + y_max)/2
            # width_n = float(x_max - x_min)
            # height_n = float(y_max - y_min)/2
            confidence = float(score[i])

            # obj_name = obj_list[int(obj_id)]
            # left, top, right, bottom = convert_yolo_coordinates_to_voc(x_c_n, y_c_n, width_n, height_n, img_width,
            #                                                            img_height)
            ## add new line to file
            # print(obj_name + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom))
            new_f.write(obj_name + " " + str(confidence) + " " + str(left) + " " + str(bottom) + " " + str(right) + " " + str(top) + '\n')


def getAnnotationInDir(dir_path):
    annotation_list = []
    for filename in glob.glob(dir_path + '/*.txt'):
        annotation_list.append(filename.split('/')[-1])
    return annotation_list


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.mode == 'train':
        imgs_path = args.data_path + "training/images/"
        labels_path = args.data_path + "training/annotations/"
    elif args.mode == "eval":
        imgs_path = args.data_path + "testing/images/"
        labels_path = args.data_path + "testing/annotations/"
    else:
        raise ValueError('mode can only be train, or eval.')

    data_transform = transforms.Compose([transforms.ToTensor()])
    #
    # print("imgs_path: ", imgs_path)
    # print("labels_path: ", labels_path)
    # print(a)
    dataset = MaskDataset(data_transform, imgs_path, labels_path, args.mode)
    print("len(dataset): ", len(dataset))
    model = get_model_instance_segmentation(3)

    if args.mode == 'train':
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)
        # Train
        num_epochs = 1000
        model.to(device)

        # parameters
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=LEARNING_RATE,
                                    momentum=0.9, weight_decay=0.0005)

        len_dataloader = len(data_loader)

        # Make folders and set parameters
        os.makedirs('../data/data2/FasterRCNN/outputs', exist_ok=True)
        os.makedirs('../data/data2/FasterRCNN/checkpoints', exist_ok=True)

        best_losses = BEST_LOSS

        for epoch in range(num_epochs):
            model.train()
            i = 0
            epoch_loss = 0
            for imgs, annotations, _ in data_loader:
                i += 1
                imgs = list(img.to(device) for img in imgs)
                annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

                batch_loss = 0
                for batch_idx in range(len(imgs)):
                    loss_dict = model([imgs[batch_idx]], [annotations[batch_idx]])
                    # print("#########################")
                    # print("loss_dict: ", loss_dict)
                    losses = sum(loss for loss in loss_dict.values())
                    batch_loss += losses

                losses = batch_loss/BATCH_SIZE

                if losses < best_losses:
                    best_losses = losses
                    torch.save(model.state_dict(),
                               '../data/data2/FasterRCNN/checkpoints/model-epoch-{}-losses-{:.8f}.pth'.format(epoch + 1, losses))
                    print('done saving')

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                print(f'Epoch: {epoch}, Batch: {i}/{len_dataloader}, Loss: {losses}')
                epoch_loss += losses
            print(epoch_loss)

    elif args.mode == 'eval':
        txt_list = getAnnotationInDir(imgs_path)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
        # Load model
        model2 = get_model_instance_segmentation(3)
        model2.load_state_dict(torch.load(args.model_path))

        model2.to(device)

        i = 0
        losses = 0
        print("len(data_loader): ", len(data_loader))
        for imgs, annotations, imgPath in data_loader:
            print("###################################")
            print("imgPath: ", imgPath)
            print("i + 842: ", i + 842)
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

            model2.train()
            with torch.no_grad():
                loss_dict = model2([imgs[0]], [annotations[0]])
                losses += sum(loss for loss in loss_dict.values())

            # print(f'Eval Loss: {losses}')

            model2.eval()
            pred2 = model2(imgs)
            print("type(): ", type(pred2))
            print("len(pred2)", len(pred2))

            print("type(pred2[0]): ", type(pred2[0]))
            print("pred2[0]: ", pred2[0])

            print("Printing generated annotations...")
            tmp_file = "../data/data2/FasterRCNN/outputs/test_output_annotations/" + imgPath[0].split('/')[-1][:-3] + "txt"
            save_prediction(pred2[0], tmp_file)
            print("Done.")

            print("Prediction")
            plot_image(imgs[0], pred2[0], "prediction_%s" % str(i+842))
            print("Target")
            plot_image(imgs[0], annotations[0], "target_%s" % str(i+842))

            i += 1

        print(f'Eval Loss: {losses/float(i)}')

    else:
        raise ValueError('mode can only be train, or eval.')




if __name__ == "__main__":
    # imgs_path = "../data/data2/images/"
    # labels_path: str = "../data/data2/annotations/"
    # imgs = list(sorted(os.listdir(imgs_path)))
    # labels = list(sorted(os.listdir(labels_path)))
    print("BATCH_SIZE: ", BATCH_SIZE)
    print("LEARNING_RATE: ", LEARNING_RATE)
    begin_time = time.time()
    main(parse_args())
    end_time = time.time()
    total_time = end_time - begin_time
    print("Total time: ", total_time)
    print("Total time in hours: ", total_time/3600)
