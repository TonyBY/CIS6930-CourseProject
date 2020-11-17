from torchvision import transforms
import os
import argparse

from MaskDetection_FasterRCNN_V0_utils import *
from MaskDataset import MaskDataset


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='A CNN based image colorizer.'
    )
    parser.add_argument('-imgs_path', default='../data/data2/images/',
                        help='data path of file containing the images')
    parser.add_argument('-labels_path', default='../data/data2/annotations/',
                        help='data path of file containing the annotations')
    parser.add_argument('-model_path', default='../data/data2/FasterRCNN/checkpoints/model-epoch-1-losses-0.017.pth/',
                        help='Pre-trained model path of FasterRCNN')
    parser.add_argument('-mode', default='train', help='Option: train/eval')

    return parser.parse_args(args)


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    imgs_path = args.imgs_path
    labels_path = args.labels_path

    data_transform = transforms.Compose([transforms.ToTensor()])

    dataset = MaskDataset(data_transform, imgs_path, labels_path)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)

    model = get_model_instance_segmentation(3)

    if args.mode == 'train':
        # Train
        num_epochs = 1
        model.to(device)

        # parameters
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)

        len_dataloader = len(data_loader)

        # Make folders and set parameters
        os.makedirs('../data/data2/FasterRCNN/outputs', exist_ok=True)
        os.makedirs('../data/data2/FasterRCNN/checkpoints', exist_ok=True)

        best_losses = 1e10

        for epoch in range(num_epochs):
            model.train()
            i = 0
            epoch_loss = 0
            for imgs, annotations in data_loader:
                i += 1
                imgs = list(img.to(device) for img in imgs)
                annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
                loss_dict = model([imgs[0]], [annotations[0]])
                losses = sum(loss for loss in loss_dict.values())

                if losses < best_losses:
                    best_losses = losses
                    torch.save(model.state_dict(),
                               '../data/data2/FasterRCNN/checkpoints/model-epoch-{}-losses-{:.3f}.pth'.format(epoch + 1, losses))
                    print('done saving')

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')
                epoch_loss += losses
            print(epoch_loss)

    elif args.mode == 'eval':
        # Load model
        model2 = get_model_instance_segmentation(3)
        model2.load_state_dict(torch.load(args.model_path))
        model2.eval()
        model2.to(device)

        for imgs, annotations in data_loader:
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            break

        pred2 = model2(imgs)

        print("Prediction")
        plot_image(imgs[2], pred2[2], "prediction")
        print("Target")
        plot_image(imgs[2], annotations[2], "target")

    else:
        raise ValueError('mode can only be train, or eval.')




if __name__ == "__main__":
    imgs_path = "../data/data2/images/"
    labels_path: str = "../data/data2/annotations/"
    # imgs = list(sorted(os.listdir(imgs_path)))
    # labels = list(sorted(os.listdir(labels_path)))
    main(parse_args())
