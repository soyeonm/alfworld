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
sys.path.append('/home/soyeonm/projects/alfworld_mrcnn')

import cv2

from alfworld.agents.detector.engine import train_one_epoch, evaluate
import alfworld.agents.detector.utils as utils
import torchvision
#from alfworld.agents.detector.mrcnn import get_model_instance_segmentation#, load_pretrained_model
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


MIN_PIXELS = 30

#OBJECTS_DETECTOR = constants.OBJECTS_DETECTOR
OBJECTS_DETECTOR = ['Bowl', 'Mug', 'Plate', 'Cup', 'Box', 'Pan', 'Pot', 'Painting', 'Television', 'HousePlant', 'Pen', 'TissueBox', 'RemoteControl', 'Pencil', 'Book', 'CellPhone', 'DeskLamp', 'Dumbbell', 'FloorLamp', 'AlarmClock', 'BaseballBat', 'Pillow', 'CD', 'TableTopDecor', 'CreditCard', 'Vase', 'Laptop', 'Newspaper', 'Plunger', 'Faucet', 'ToiletPaper', 'DishSponge', 'GarbageBag', 'Statue', 'SprayBottle', 'Apple', 'Ladle', 'Bottle', 'Spatula', 'SoapBottle', 'SaltShaker', 'Watch', 'KeyChain', 'Tomato', 'Lettuce', 'Potato', 'Bread', 'ButterKnife', 'Candle', 'PepperShaker', 'Spoon', 'Kettle', 'SoapBar', 'BasketBall', 'Knife', 'Fork', 'PaperTowelRoll', 'Boots', 'Cloth', 'TennisRacket', 'Egg', 'WineBottle', 'TeddyBear', 'VacuumCleaner', 'RoomDecor', 'Desktop']
STATIC_RECEPTACLES = ['Desk', 'ArmChair', 'Chair', 'Safe', 'Dresser', 'Bed', 'GarbageCan', 'DiningTable', 'SideTable',  'Cart', 'LaundryHamper', 'Sink', 'Toilet', 'ShelvingUnit', 'Fridge',  'Toaster', 'CoffeeMachine', 'CounterTop', 'Floor', 'DogBed', 'TVStand',  'Sofa', 'Stool',  'WashingMachine', 'CoffeeTable', 'ClothesDryer', 'Microwave', 'Ottoman']
#STATIC_RECEPTACLES = constants.STATIC_RECEPTACLES
ALL_DETECTOR = set(OBJECTS_DETECTOR + STATIC_RECEPTACLES + ['Ceiling_room', 'room', 'window', 'door', 'wall'])
#recep_path = os.environ['RECEP_PATH']
#obj_path = os.environ['OBJ_PATH']

#object_detector_objs = ['AlarmClock', 'Apple', 'AppleSliced', 'BaseballBat', 'BasketBall', 'Book', 'Bowl', 'Box', 'Bread', 'BreadSliced', 'ButterKnife', 'CD', 'Candle', 'CellPhone', 'Cloth', 'CreditCard', 'Cup', 'DeskLamp', 'DishSponge', 'Egg', 'Faucet', 'FloorLamp', 'Fork', 'Glassbottle', 'HandTowel', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle', 'Laptop', 'LaundryHamperLid', 'Lettuce', 'LettuceSliced', 'LightSwitch', 'Mug', 'Newspaper', 'Pan', 'PaperTowel', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate', 'Plunger', 'Pot', 'Potato', 'PotatoSliced', 'RemoteControl', 'SaltShaker', 'ScrubBrush', 'ShowerDoor', 'SoapBar', 'SoapBottle', 'Spatula', 'Spoon', 'SprayBottle', 'Statue', 'StoveKnob', 'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'ToiletPaper', 'ToiletPaperRoll', 'Tomato', 'TomatoSliced', 'Towel', 'Vase', 'Watch', 'WateringCan', 'WineBottle']

# alfworld_receptacles = [
#         'BathtubBasin',
#         'Bowl',
#         'Cup',
#         'Drawer',
#         'Mug',
#         'Plate',
#         'Shelf',
#         'SinkBasin',
#         'Box',
#         'Cabinet',
#         'CoffeeMachine',
#         'CounterTop',
#         'Fridge',
#         'GarbageCan',
#         'HandTowelHolder',
#         'Microwave',
#         'PaintingHanger',
#         'Pan',
#         'Pot',
#         'StoveBurner',
#         'DiningTable',
#         'CoffeeTable',
#         'SideTable',
#         'ToiletPaperHanger',
#         'TowelHolder',
#         'Safe',
#         'BathtubBasin',
#         'ArmChair',
#         'Toilet',
#         'Sofa',
#         'Ottoman',
#         'Dresser',
#         'LaundryHamper',
#         'Desk',
#         'Bed',
#         'Cart',
#         'TVStand',
#         'Toaster',
# ]



def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    anchor_generator = AnchorGenerator(
        sizes=tuple([(4, 8, 16, 32, 64, 128, 256, 512) for _ in range(5)]),
        aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]))
    model.rpn.anchor_generator = anchor_generator

    # 256 because that's the number of features that FPN returns
    model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def load_pretrained_model(device):
    if args.object_types == "objects":
        categories = len(OBJECTS_DETECTOR)
        #path = obj_path
    elif args.object_types =="receptacles":
        categories = len(STATIC_RECEPTACLES)
        #path = recep_path
    #print("path is ", path)
    mask_rcnn = get_model_instance_segmentation(categories+1)
    #pickle.dump(torch.load(path, map_location=device), open("loaded.p", "wb"))
    #mask_rcnn.load_state_dict(torch.load(path, map_location=device))
    print("LOADED MODELS!")
    return mask_rcnn


def get_object_classes(object_type):
    if object_type == "objects":
        return OBJECTS_DETECTOR
    elif object_type == "receptacles":
        return STATIC_RECEPTACLES
    else:
        return ALL_DETECTOR

class AlfredDataset(object):
    def __init__(self, root, transforms, args, train_dataset):
        self.root = root
        self.transforms = transforms
        self.args = args
        self.object_classes = {v:i for i, v in enumerate(get_object_classes(args.object_types))}

        # load all image files, sorting them to
        # ensure that they are aligned
        self.get_data_files(root, train_dataset=train_dataset)


    def png_only(self, file_list):
        return [f for f in file_list if f[-4:] == '.png']


    # def get_data_files_teach(self, root, balance_scenes=False, train_dataset=False):
    #     kitchen_path = os.path.join(root, 'kitchen')#, 'images')
    #     living_path = os.path.join(root, 'living')# 'images')
    #     bedroom_path = os.path.join(root, 'bedroom')#, 'images')
    #     bathroom_path = os.path.join(root, 'bathroom')#, 'images')



    #     kitchen = glob(kitchen_path + '/*/images/*.png')
    #     living = glob(living_path + '/*/images/*.png')
    #     bedroom = glob(bedroom_path + '/*/images/*.png')
    #     bathroom = glob(bathroom_path + '/*/images/*.png')


    #     #if self.args.balance_scenes or not(train_dataset):
    #     min_size = int(len(kitchen)/4)
    #     #kitchen = [k for i,k in enumerate(kitchen) if i % self.args.kitchen_sample_factor == 0]

    #     #living = [k for i,k in enumerate(living) if i %self.args.living_sample_factor == 0]

    #     if not(train_dataset):
    #         #just keep 1000
    #         print("Total is ", len(kitchen+living + bedroom + bathroom))
    #         ka = int(len(kitchen) / 100)
    #         la = int(len(living) / 100)
    #         bea = int(len(bedroom)/ 10)
    #         baa = int(len(bathroom)/10)
    #         kitchen = [k for i,k in enumerate(kitchen) if i % ka == 0]
    #         living = [k for i,k in enumerate(living) if i % la == 0]
    #         bedroom = [k for i,k in enumerate(bedroom) if i % bea == 0]
    #         bathroom = [k for i,k in enumerate(bathroom) if i % baa == 0]
    #         print("Total after is ", len(kitchen+living + bedroom + bathroom))


    #     self.imgs = kitchen + living + bedroom + bathroom
    #     self.masks = [f.replace("/images/", "/masks/") for f in self.imgs]
    #     self.metas = [f.replace("/images/", "/meta/").replace(".png", ".json") for f in self.imgs]

    def get_data_files(self, root, train_dataset=False, select_num=20):
        

        #self.imgs = glob(os.path.join(root, '*', 'rgb', '*.png'))
        #self.masks = glob(os.path.join(root, '*', 'masks', '*.p'))
        #Just select 10 each
        imgs = []
        for g in glob(os.path.join(root, '*', 'rgb')):
            #Just add the first 10 
            imgs += glob(g + '/*')[:select_num]
        masks = [i.replace('rgb', 'masks').replace('.png', '.p') for i in imgs] 

        self.imgs = []; self.masks = []
        breakpoint()
        for img, mask in zip(imgs, masks):
            if os.path.exists(mask):
                self.imgs.append(img); self.masks.append(mask) 

        #self.metas = [os.path.join(root, "meta", f) for f in list(sorted(os.listdir(os.path.join(root, "meta"))))]

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        img = Image.open(img_path).convert("RGB")

        mask_pickle = pickle.load(open(mask_path, 'rb'))
        masks, boxes, labels = [], [], []

        for object_id, tf_mask in mask_pickle.items():
            object_class = object_id.split("|", 1)[0] if "|" in object_id else ""
            if "Basin" in object_id:
                object_class += "Basin"
            #TODO: DELTE THIS
            if not(object_class in self.object_classes):
                if not(object_class in ALL_DETECTOR):
                    print("Object class ", object_class, " Not in ALL_DETECTOR ")
            else:
                class_idx = self.object_classes[object_class]
                #smask = torch.tensor(tf_mask).byte()

                pos = np.where(tf_mask)
                num_pixels = len(pos[0])

                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])

                # skip if not sufficient pixels
                # if num_pixels < MIN_PIXELS:
                if (xmax-xmin)*(ymax-ymin) < MIN_PIXELS:
                    continue

                masks.append(tf_mask)
                #pickle.dump(masks, open("masks_" + str(idx) + ".p", "wb"))
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_idx)

        if len(boxes) == 0:
            return None, None

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


    #Old
    # def __getitem__(self, idx):
    #     # load images ad masks
    #     img_path = self.imgs[idx]
    #     mask_path = self.masks[idx]
    #     #meta_path = self.metas[idx]

    #     #print("Opening: %s" % (self.imgs[idx]))

    #     #with open(meta_path, 'r') as f:
    #     #    color_to_object = json.load(f)

    #     #print("img_path is", img_path)
        
    #     img = Image.open(img_path).convert("RGB")
    #     if self.args.sanity_check:
    #         cv2.imwrite("real_training_img.png", img)
    #     if args.resize:
    #         img.resize((300,300))
            
    #     # note that we haven't converted the mask to RGB,
    #     # because each color corresponds to a different instance
    #     # with 0 being background
    #     mask = Image.open(mask_path)
    #     if args.resize:
    #         mask.resize((300,300))

    #     mask = np.array(mask)
    #     #print("mask_path is", mask_path)
    #     im_width, im_height = mask.shape[0], mask.shape[1]
    #     seg_colors = np.unique(mask.reshape(im_height*im_height, 3), axis=0)

    #     masks, boxes, labels = [], [], []
    #     for color in seg_colors:
    #         color_str = str(tuple(color[::-1]))
    #         if color_str in color_to_object:
    #             object_id = color_to_object[color_str]
    #             object_class = object_id.split("|", 1)[0] if "|" in object_id else ""
    #             if "Basin" in object_id:
    #                 object_class += "Basin"
    #             if object_class in self.object_classes:
    #                 smask = np.all(mask == color, axis=2)
    #                 pos = np.where(smask)
    #                 num_pixels = len(pos[0])

    #                 xmin = np.min(pos[1])
    #                 xmax = np.max(pos[1])
    #                 ymin = np.min(pos[0])
    #                 ymax = np.max(pos[0])

    #                 # skip if not sufficient pixels
    #                 # if num_pixels < MIN_PIXELS:
    #                 if (xmax-xmin)*(ymax-ymin) < MIN_PIXELS:
    #                     continue

    #                 class_idx = self.object_classes.index(object_class)

    #                 masks.append(smask)
    #                 #pickle.dump(masks, open("masks_" + str(idx) + ".p", "wb"))
    #                 boxes.append([xmin, ymin, xmax, ymax])
    #                 labels.append(class_idx)

    #                 if self.args.debug:
    #                     disp_img = np.array(img)
    #                     cv2.rectangle(disp_img, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
    #                     cv2.putText(disp_img, object_class, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
    #                     sg = np.uint8(smask[:, :, np.newaxis])*255

    #                     print(xmax-xmin, ymax-ymin, num_pixels)
    #                     cv2.imshow("img", np.array(disp_img))
    #                     cv2.imshow("sg", sg)
    #                     cv2.waitKey(0)

    #     if len(boxes) == 0:
    #         return None, None

    #     iscrowd = torch.zeros(len(masks), dtype=torch.int64)
    #     boxes = torch.as_tensor(boxes, dtype=torch.float32)
    #     labels = torch.as_tensor(labels, dtype=torch.int64)
    #     masks = torch.as_tensor(masks, dtype=torch.uint8)

    #     image_id = torch.tensor([idx])
    #     area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

    #     target = {}
    #     target["boxes"] = boxes
    #     target["labels"] = labels
    #     target["masks"] = masks
    #     target["image_id"] = image_id
    #     target["area"] = area
    #     target["iscrowd"] = iscrowd

    #     if self.transforms is not None:
    #         img, target = self.transforms(img, target)

    #     return img, target

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
    num_classes = len(get_object_classes(args.object_types))+1
    # use our dataset and defined transformations
    dataset = AlfredDataset(args.data_path, get_transform(train=True), args, True)
    dataset_test = AlfredDataset(args.test_data_path, get_transform(train=False), args, False)

    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    indices = list(range(len(dataset)))
    indices_test = list(range(len(dataset_test)))
    # if not(args.without_40):
    #     print("using 40 for validation")
    #     dataset = torch.utils.data.Subset(dataset, indices[:-40])
    #     #pickle.dump(dataset, open("dataset_train.p", "wb"))
    #     dataset_test = torch.utils.data.Subset(dataset_test, indices_test[-40:])
    # else:
    dataset = torch.utils.data.Subset(dataset, indices)
    dataset_test = torch.utils.data.Subset(dataset_test, indices_test)

    breakpoint()
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # get the model using our helper function

    #if len(args.load_model) > 0:
    model = load_pretrained_model(device)
    #else:
    #    model = get_model_instance_segmentation(num_classes, args.backbone)

    # move model to the right device
    model.to(device)

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
    #print("Starting sanity evaluation")
    #if args.evaluate:
    #    epoch = -1
    #    if not(args.no_logs):
    #        log_file, old_out = start_write_log_sys(log_name)
    #    c, logs = evaluate(model, data_loader_test, device=device, epoch=epoch)
    #    del c
    #    if not(args.no_logs):
    #        end_write_log_sys(log_file, old_out)

    for epoch in range(num_epochs):
        print("Epoch ", epoch, "starting!")
        # train for one epoch, printing every 10 iterations
        if not(args.no_logs):
            log_file, old_out = start_write_log_sys(log_name)
            print("log done!")
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=500)
        # update the learning rate
        lr_scheduler.step()
        # # evaluate on the test dataset
        model_path = os.path.join(args.save_path, "%s_%03d.pth" % (args.save_name, epoch))
        torch.save(model.state_dict(), model_path)
        if args.evaluate:
            c, logs = evaluate(model, data_loader_test, device=device, epoch=epoch)
            del c 
        # save model
        
        print("Saving %s" % model_path)
        if not(args.no_logs):
            end_write_log_sys(log_file, old_out)

    print("Done training!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_logs", action='store_true')
    parser.add_argument("--backbone", type=int, default=50)
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--resize", action='store_true')

    parser.add_argument("--without_40", action = "store_true")
    parser.add_argument("--save_path", type=str, default="data/")
    parser.add_argument("--object_types", choices=["objects", "receptacles", "all"], default="all")
    parser.add_argument("--save_name", type=str, default="mrcnn_alfred_objects")
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)

    # parser.add_argument("--kitchen_sample_factor", type=int, default=35)
    # parser.add_argument("--living_sample_factor", type=int, default=4)
    # parser.add_argument("--balance_scenes", action='store_true')
    # parser.add_argument("--kitchen_factor", type=float, default=1.0)
    # parser.add_argument("--living_factor", type=float, default=1.0)
    # parser.add_argument("--bedroom_factor", type=float, default=1.0)
    # parser.add_argument("--bathroom_factor", type=float, default=1.0)

    parser.add_argument("--num_epochs", type=int, default=10)

    parser.add_argument("--sanity_check", action = "store_true")

    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(args)
