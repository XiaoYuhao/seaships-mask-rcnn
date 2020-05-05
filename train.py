import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import functional as F
import numpy as np 
import os
from PIL import Image
from utils import readxml, TensorToPIL
from faster_rcnn import fasterrcnn_resnet50_fpn
from mask_rcnn import maskrcnn_resnet50_fpn
from dataset import SeashipDataset
from torchvision import transforms
import utils
import logger
import traceback

logger = logger.getLogger(log_name="mask_rcnn_weakly_5_2_002")

labels_dict = {
            'background' : 0,
            'ore carrier' : 1,
            'passenger ship' : 2,
            'container ship' : 3,
            'bulk cargo carrier' : 4,
            'general cargo ship' : 5,
            'fishing boat' : 6
        }
classes_name =['background','ore carrier','passenger ship','container ship','bulk cargo carrier','general cargo ship','fishing boat']

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    img_dir = list(sorted(os.listdir('../SeaShips/JPEGImages')))
    anno_dir = list(sorted(os.listdir('../SeaShips/Annotations')))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    num_epochs = 5
    batch_size = 2

    model.to(device)
    model.train()

    loss_sum = 0

    for epoch in range(num_epochs):
        for idx in range(0,5000,batch_size):
            img_path = [os.path.join("../SeaShips/JPEGImages", img_dir[i]) for i in range(idx,idx+batch_size)]
            anno_path = [os.path.join("../SeaShips/Annotations", anno_dir[i]) for i in range(idx,idx+batch_size)]
            img = [Image.open(path).convert('RGB') for path in img_path]
            targets = [readxml(path,labels_dict,device) for path in anno_path]
            img_var = [F.to_tensor(image).to(device) for image in img]
            #img = torch.stack(img_var, dim=0).to(device)
            loss_dict = model.forward(img_var, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_sum += losses

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if idx % 10 == 0:
                print("[%d]rpn_loss: %f" %(idx, loss_sum))
                loss_sum = 0
    
    torch.save(model.state_dict(), "./faster_rcnn_1.pth")

def test():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.load_state_dict(torch.load("./faster_rcnn_1.pth"))
    img_dir = list(sorted(os.listdir('../SeaShips/JPEGImages')))
    anno_dir = list(sorted(os.listdir('../SeaShips/Annotations')))

    model.to(device)
    model.eval()

    img_path = os.path.join("../SeaShips/JPEGImages/", img_dir[1000])
    img = [Image.open(img_path).convert('RGB')]
    img_var = [F.to_tensor(image).to(device) for image in img]
    result = model.forward(img_var)
    print(result)
    boxes = result[0]['boxes'].cpu().detach().numpy()
    scores = result[0]['scores'].cpu().detach().numpy()
    print(boxes)
    index = np.where(scores > 0.9)
    boxes = boxes[index]
    image = img[0]
    utils.drawRectangle(image, boxes)
    image.save("./result/res5.jpg")

def read_mask(path):
    mask = Image.open(path).convert("RGB")
    mask = np.array(mask)
    masks = np.zeros()

def test_mask():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 7)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 7)

    model.load_state_dict(torch.load("./mask_rcnn_2.pth"))
    model.to(device)
    #model.train()
    model.eval()
    data = SeashipDataset("../SeaShips", None)

    targets = []

    image, target = data.__getitem__(1888)
    img_var = [F.to_tensor(image).to(device)]
    original_image_sizes = [img.shape[-2:] for img in img_var]
    targets.append(target)
    targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
    #result = model.forward(img_var, targets)
    result = model.forward(img_var)
    '''
    batch_size = 2

    label_struct = dict()

    for idx in range(5000,7000,batch_size):
        imgs = []
        targets = []
        for i in range(idx, idx+batch_size):
            img, target = data.__getitem__(i)
            imgs.append(F.to_tensor(img).to(device))
            target = {k : v.to(device) for k, v in target.items()}
            targets.append(target)
            
        result = model.forward(imgs)

        for res in result:
            bbox

    '''

    
    print(result)
    print(target['labels'])
    boxes = result[0]['boxes'].cpu().detach().numpy()
    scores = result[0]['scores'].cpu().detach().numpy()
    masks = result[0]['masks'].cpu()

    index = np.where(scores > 0.9)
    boxes = boxes[index]
    masks = masks[index]
    masks= torch.where(masks > 0.5, torch.full_like(masks, 1), torch.full_like(masks, 0))
    m = torch.zeros(original_image_sizes[0])
    print(m.shape)
    for mask in masks:
        m += mask[0]
    m = torch.where(m > 0.5, torch.full_like(m, 1), torch.full_like(m, 0))
    img_mask = TensorToPIL(m)
    img_mask.save("./result/res11.png")
    #masks[pos] = 1
    #print(boxes)
    #print(masks[0][0][int(boxes[0][1]):int(boxes[0][3]),int(boxes[0][0]):int(boxes[0][2])])
    '''
    image = image.convert("RGBA")
    for mask in masks:
        mask = TensorToPIL(mask)
        mask.convert("RGBA")
        image = drawMasks(image, mask)
    drawRectangle(image, boxes)
    image.save("./result/res9.png")
    '''

def train_mask():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 7)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 7)

    model.to(device)
    model.train()

    data = SeashipDataset("../../SeaShips", None)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    num_epochs = 5
    batch_size = 2

    loss_sum = 0
    loss_classifier = 0
    loss_box_reg = 0
    loss_mask = 0

    with open("./train_data", "r") as f:
        lines = f.readlines()
        train_list = [int(line) for line in lines]
    
    print(len(train_list))

    for epoch in range(num_epochs):
        lr_scheduler = None
        if epoch == 0:
            warmup_factor = 1./1000
            warmup_iters = min(1000, 5000 - 1)
            lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

        for idx in range(0,len(train_list),batch_size):
            try:
                imgs = []
                targets = []
                for i in range(idx, idx+batch_size):
                    img, target = data.__getitem__(train_list[i]-1)
                    imgs.append(F.to_tensor(img).to(device))
                    target = {k : v.to(device) for k, v in target.items()}
                    targets.append(target)

                loss_dict = model.forward(imgs, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_sum += losses

                
                #loss_classifier += loss_dict['loss_classifier'].values()
                #loss_box_reg += loss_dict['loss_box_reg'].values()
                #loss_mask += loss_dict['loss_mask'].values()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                if lr_scheduler is not None:
                    lr_scheduler.step()
            except:
                logger.error(str(traceback.format_exc()))


            if idx % 12 == 0:
                logger.debug("[%d]total_loss: %f" %(idx, loss_sum))
                #logger.debug("[%d]loss: %f loss_classifier: %f loss_box_reg: %f loss_mask: %f" %(idx, loss_sum, loss_classifier, loss_box_reg, loss_mask))
                loss_sum = 0
                #loss_classifier = 0
                #loss_box_reg = 0
                #loss_mask = 0
    
    torch.save(model.state_dict(), "../model/mask_rcnn_5_2_002.pth")
    logger.debug("train successfully!")\

def res_mask():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 7)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 7)

    model.load_state_dict(torch.load("../model/mask_rcnn_10_2_005.pth"))
    model.to(device)
    model.eval()

    data = SeashipDataset("../SeaShips", None)

    with open("./train_data", "r") as f:
        lines = f.readlines()
        train_list = [int(line) for line in lines]
    
    print(len(train_list))

    batch_size = 2

    for idx in range(0,len(train_list),batch_size):
        imgs = []
        for i in range(idx, idx+batch_size):
            img, target = data.__getitem__(train_list[i])
            imgs.append(F.to_tensor(img).to(device))
        
        original_image_sizes = [img.shape[-2:] for img in imgs]
        result = model.forward(imgs)

        for j, res in enumerate(result):
            scores = res['scores'].cpu().detach().numpy()
            masks = res["masks"].cpu()
            index = np.where(scores > 0.9)                  #只要9分以上的
            masks = masks[index]
            masks= torch.where(masks > 0.5, torch.full_like(masks, 1), torch.full_like(masks, 0))
            m = torch.zeros(original_image_sizes[0])
            for mask in masks:
                m += mask[0]
            m = torch.where(m > 0.5, torch.full_like(m, 1), torch.full_like(m, 0))
            img_mask = TensorToPIL(m)
            mask_name = "{:0>6d}.png".format(idx+j+1)
            path = os.path.join("./res_mask/",mask_name)
            img_mask.save(path)
            print(path)

    #torch.save(model.state_dict(), "./mask_rcnn_2.pth")

def test_train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 7)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 7)

    model.load_state_dict(torch.load("./mask_rcnn_2.pth"))
    model.to(device)
    model.train()

    data = SeashipDataset("../SeaShips", None)

    imgs = []
    targets = []

    for i in range(2):
        img, target = data.__getitem__(i)
        imgs.append(F.to_tensor(img).to(device))
        target = {k : v.to(device) for k, v in target.items()}
        targets.append(target)

    targets.append(target)
    targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
    result = model.forward(imgs, targets)


def weakly_supervision_train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 7)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 7)

    model.load_state_dict(torch.load("../model/mask_rcnn_5_2_002.pth"))  #加载已训练的模型
    model.to(device)
    model.open_weakly_supervision_train()                   #打开弱监督训练方式

    for name, params in model.named_parameters():
        if 'mask' not in name:                              #冻结mask分支以外的参数
            params.requires_grad = False

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    num_epochs = 5
    batch_size = 2

    loss_sum = 0

    data = SeashipDataset("../../SeaShips", None)

    with open("./train_data", "r") as f:
        lines = f.readlines()
        train_list = [int(line) for line in lines]
    
    print(len(train_list))

    for epoch in range(num_epochs):
        for idx in range(0,len(train_list),batch_size):
            try:
                imgs = []
                targets = []
                for i in range(idx, idx+batch_size):
                    img, target = data.getitem2(train_list[i]-1, epoch)
                    imgs.append(F.to_tensor(img).to(device))
                    target = {k : v.to(device) for k, v in target.items()}
                    targets.append(target)

                original_image_sizes = [img.shape[-2:] for img in imgs]
                loss_dict, result = model.forward(imgs, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_sum += losses

                #print(result)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                for j, res in enumerate(result):
                    scores = res['scores'].cpu().detach().numpy()
                    masks = res["masks"].cpu()
                    #print(masks[0].shape)
                    index = np.where(scores > 0.9)                  #只要9分以上的
                    masks = masks[index]
                    masks= torch.where(masks > 0.5, torch.full_like(masks, 1), torch.full_like(masks, 0))
                    m = torch.zeros(original_image_sizes[0])
                    for mask in masks:
                        m += mask[0]
                    m = torch.where(m > 0.5, torch.full_like(m, 1), torch.full_like(m, 0))
                    img_mask = TensorToPIL(m)
                    data.updatemask(idx+j, img_mask, epoch)
            except:
                logger.error(str(traceback.format_exc()))

            if idx % 10 == 0:
                #print("[%d]rpn_loss: %f" %(idx, loss_sum))
                logger.debug("[%d]total_loss: %f" %(idx, loss_sum))
                loss_sum = 0
    
    torch.save(model.state_dict(), "./mask_rcnn_weakly_5_2_002.pth")


from faster_rcnn import FastRCNNPredictor
from mask_rcnn import MaskRCNNPredictor

if __name__ == '__main__':
    #train()
    #test()
    #test_mask()
    #train_mask()
    #res_mask()
    #test_train()
    weakly_supervision_train()
