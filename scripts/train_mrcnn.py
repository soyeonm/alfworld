# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import random
import json
import argparse
from PIL import Image, ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True
import pickle
import sys
sys.path.append('/home/soyeonm/projects/devendra/alfworld/alfworld')

import cv2

from alfworld.agents.detector.engine import train_one_epoch, evaluate
import alfworld.agents.detector.utils as utils
import torchvision
from alfworld.agents.detector.mrcnn import get_model_instance_segmentation1 #, load_pretrained_model
import alfworld.agents.detector.transforms as T

import alfworld.gen.constants as constants
import pickle
import sys

from glob import glob

import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import pickle

import os
import sys


MIN_PIXELS = 100 

small_objects =  ['basket', 'book', 'bowl', 'cup', 'hat', 'plate', 'shoe', 'stuffed_toy']
small_objects_cat_to_idx = {v:k for k,v in enumerate(small_objects)}
#{'basket': 0, 'book': 1, 'bowl': 2, 'cup': 3, 'hat': 4, 'plate': 5, 'shoe': 6, 'stuffed_toy': 7}

# def get_object_classes(object_type):
#     #TODO
#     if object_type == "objects":
#         return OBJECTS_DETECTOR
#     elif object_type == "receptacles":
#         return STATIC_RECEPTACLES
#     else:
#         return ALL_DETECTOR

class AlfredDataset(object):
    def __init__(self, root, transforms, args, train_dataset):
        self.root = root
        self.transforms = transforms
        self.args = args
        self.object_classes = small_objects_cat_to_idx

        # load all image files, sorting them to
        # ensure that they are aligned
        self.get_data_files(root, train_dataset=train_dataset)


    def png_only(self, file_list):
        return [f for f in file_list if f[-4:] == '.png']

    def get_data_files(self, root, balance_scenes=False, train_dataset=False):
        rgb_dirs = sorted(glob(root + "/*/rgb_*/*"))
        mask_dirs = [rgb_path.replace('/rgb_', '/masks_').replace('.png', '.p') for rgb_path in rgb_dirs]
        #self.imgs = glob(root + "/*/rgb_*/*")
        #self.masks = [rgb_path.replace('/rgb_', '/masks_').replace('.png', '.p') for rgb_path in self.imgs]

        self.imgs = []
        self.masks = []
        for rgb, mask in zip(rgb_dirs, mask_dirs):
            if os.path.exists(rgb) and os.path.exists(mask):
                self.imgs.append(rgb)
                self.masks.append(mask) 

         

    # def get_data_files(self, root, balance_scenes=False, train_dataset=False):
    #     if balance_scenes:
    #         kitchen_path = os.path.join(root, 'kitchen', 'images')
    #         living_path = os.path.join(root, 'living', 'images')
    #         bedroom_path = os.path.join(root, 'bedroom', 'images')
    #         bathroom_path = os.path.join(root, 'bathroom', 'images')

    #         kitchen = self.png_only(list(sorted(os.listdir(kitchen_path))))
    #         living = self.png_only(list(sorted(os.listdir(living_path))))
    #         bedroom = self.png_only(list(sorted(os.listdir(bedroom_path))))
    #         bathroom = self.png_only(list(sorted(os.listdir(bathroom_path))))


    #         min_size = min(len(kitchen), len(living), len(bedroom), len(bathroom))
    #         kitchen = [os.path.join(kitchen_path, f) for f in random.sample(kitchen, int(min_size*self.args.kitchen_factor))]
    #         living = [os.path.join(living_path, f) for f in random.sample(living, int(min_size*self.args.living_factor))]
    #         bedroom = [os.path.join(bedroom_path, f) for f in random.sample(bedroom, int(min_size*self.args.bedroom_factor))]
    #         bathroom = [os.path.join(bathroom_path, f) for f in random.sample(bathroom, int(min_size*self.args.bathroom_factor))]

    #         self.imgs = kitchen + living + bedroom + bathroom
    #         self.masks = [f.replace("/images/", "/masks/") for f in self.imgs]
    #         self.metas = [f.replace("/images/", "/meta/").replace(".png", ".json") for f in self.imgs]

    #     else:
    #         self.imgs = [os.path.join(root, "images", f) for f in list(sorted(os.listdir(os.path.join(root, "images"))))]
    #         self.masks = [os.path.join(root, "masks", f) for f in list(sorted(os.listdir(os.path.join(root, "masks"))))]
    #         self.metas = [os.path.join(root, "meta", f) for f in list(sorted(os.listdir(os.path.join(root, "meta"))))]

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        #meta_path = self.metas[idx]

        #print("Opening: %s" % (self.imgs[idx]))

        # with open(meta_path, 'r') as f:
        #     color_to_object = json.load(f)

        #print("img_path is", img_path)
        
        img = Image.open(img_path).convert("RGB")
        if args.resize:
            img.resize((300,300))
            
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = pickle.load(open(mask_path, 'rb'))
        mask =np.transpose(mask, (2, 0, 1))
        if args.resize:
            mask.resize((300,300))

        #mask = np.array(mask)
        #print("mask_path is", mask_path)
        im_width, im_height = mask.shape[1], mask.shape[2]
        #seg_colors = np.unique(mask.reshape(im_height*im_height, 3), axis=0)

        masks, boxes, labels = [], [], []
        for i in range(mask.shape[0]):
            if np.sum(mask[i]) >=100:
                class_idx = i #self.object_classes[object_class]
                smask = mask[i] ==1
                pos = np.where(smask)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                masks.append(smask)
                #pickle.dump(masks, open("masks_" + str(idx) + ".p", "wb"))
                object_class = small_objects[i]
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_idx)

                if self.args.debug:
                    disp_img = np.array(img)
                    cv2.rectangle(disp_img, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
                    cv2.putText(disp_img, object_class, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
                    sg = np.uint8(smask[:, :, np.newaxis])*255

                    if not(os.path.exists('debug')):
                        os.makedirs('debug')
                    print(xmax-xmin, ymax-ymin)#, num_pixels)
                    cv2.imwrite("debug/img_i.png", np.array(disp_img))
                    cv2.imwrite("debug/sg_i.png", sg)
                    #cv2.waitKey(0) 

        if len(boxes) == 0:
            return None, None
        else:
            masks = np.stack(masks, axis=0)*1.0
        #breakpoint()
        iscrowd = torch.zeros(len(masks), dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def start_write_log_sys(log_file_name):
    old_stdout = sys.stdout

    log_file = open(log_file_name,"a")

    sys.stdout = log_file

    return log_file, old_stdout

def end_write_log_sys(log_file, old_stdout):

    sys.stdout = old_stdout

    log_file.close()

def write_log(logs, log_name):
    f = open(log_name , "a")
    for log in logs:
        f.write(log + "\n")
    f.close()

def main(args):
    # train on the GPU or on the CPU, if a GPU is not available
    #torch.device("cuda:" + str(self.args.sem_seg_gpu) if args.cuda else "cpu")
    if not os.path.exists(os.path.join(args.save_path, "logs")):
        os.makedirs(os.path.join(args.save_path, "logs"))

    log_name = os.path.join(args.save_path, "logs", args.save_name + "_log.txt")

    device = torch.device('cuda:' + str(args.gpu_num)) if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = len(small_objects)+1
    # use our dataset and defined transformations
    dataset = AlfredDataset(args.data_path, get_transform(train=True), args, True)
    dataset_test = AlfredDataset(args.test_data_path, get_transform(train=False), args, False)

    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    indices = list(range(len(dataset)))
    indices_test = list(range(len(dataset_test)))
    if not(args.without_40):
        print("using 40 for validation")
        dataset = torch.utils.data.Subset(dataset, indices[:-40])
        #pickle.dump(dataset, open("dataset_train.p", "wb"))
        dataset_test = torch.utils.data.Subset(dataset_test, indices_test[-40:])
    else:
        dataset = torch.utils.data.Subset(dataset, indices)
        dataset_test = torch.utils.data.Subset(dataset_test, indices_test)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # get the model using our helper function

    #if len(args.load_model) > 0:
    model = get_model_instance_segmentation1(len(small_objects)+1, backbone = args.backbone).to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = args.num_epochs

    for epoch in range(num_epochs):
        print("Epoch ", epoch, "starting!")
        # train for one epoch, printing every 10 iterations
        if not(args.no_logs):
            log_file, old_out = start_write_log_sys(log_name)
            print("log done!")
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # # evaluate on the test dataset
        
        if epoch %5 ==0 and args.evaluate:
            model_path = os.path.join(args.save_path, "%s_%03d.pth" % (args.save_name, epoch))
            torch.save(model.state_dict(), model_path)
            c, logs = evaluate(model, data_loader_test, device=device, epoch=epoch)
            del c 
        # save model
        
        print("Saving %s" % model_path)
        if not(args.no_logs):
            end_write_log_sys(log_file, old_out)

    print("Done training!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--no_logs", action='store_true')
    parser.add_argument("--backbone", type=int, default=50)
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--resize", action='store_true')

    parser.add_argument("--save_path", type=str, default="data/")
    parser.add_argument("--save_name", type=str, default="mrcnn_alfred_objects")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--without_40", action = "store_true")

    parser.add_argument("--sanity_check", action = "store_true")
    parser.add_argument("--no_logs", action = "store_true")

    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(args)
