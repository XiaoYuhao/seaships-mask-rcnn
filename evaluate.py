import numpy as np
import torch
from torchvision.transforms import functional as F
from utils import readxml, TensorToPIL, drawMasks, drawRectangle
from mask_rcnn import maskrcnn_resnet50_fpn, MaskRCNNPredictor
from faster_rcnn import FastRCNNPredictor
from dataset import SeashipDataset
import os
import traceback

def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def get_TP_FP_FN(dts, gts, iou_thre):
    '''
    传入一张图片某一类的检测结果
    '''
    TP = 0
    FP = 0
    FN = 0
    dts = torch.from_numpy(dts)
    gts = torch.from_numpy(gts)
    iou = box_iou(dts, gts) #[N, M]
    #print(iou)
    has_been_detection = []
    for i in range(len(dts)):
        flag = False
        for j in range(len(gts)):
            if j in has_been_detection:         #第j个gt已经被成功检测
                continue
            if iou[i, j] > iou_thre:            #iou大于阈值，检测成功
                TP += 1
                flag = True                 
                has_been_detection.append(j)
        if flag == False:                       #若某个dt与所有的gts的iou值都没有大于阈值，则为FP
            FP += 1
    FN = len(gts) - len(has_been_detection)     #遍历完成后，gts中剩下的没有匹配上的就是FN
    return TP, FP, FN

class Evaluator(object):
    def __init__(self, num_class, iou_thre):
        self.num_class = num_class
        self.TP = np.zeros(num_class,dtype=np.int32)
        self.FP = np.zeros(num_class,dtype=np.int32)
        self.FN = np.zeros(num_class,dtype=np.int32)
        self.iou_thre = iou_thre
        #for i in range(num_class+1):
        #    self.TP[i] = 0
        #    self.FP[i] = 0
        #    self.FN[i] = 0
    
    def get_label(self, result, target):
        labels = result['labels'].cpu().detach().numpy()
        #print(len(labels))
        boxes = result['boxes'].cpu().detach().numpy()
        scores = result['scores'].cpu().detach().numpy()
        label_dict = dict()
        for label,box,score in zip(labels, boxes, scores):
            if label not in label_dict.keys():
                label_dict[label] = list()
            obj = dict()
            obj['label'] = label
            obj['score'] = score
            obj['box'] = box
            label_dict[label].append(obj)

        target_labels = target['labels'].cpu().detach().numpy()
        target_boxes = target['boxes'].cpu().detach().numpy()
        target_dict = dict()
        for label,box in zip(target_labels, target_boxes):
            if label not in target_dict.keys():
                target_dict[label] = list()
            target_dict[label].append(box)

        #print(label_dict.keys())
        #print(label_dict.values())
        #print(target_dict.keys())
        #print(target_dict.values())

        for label in label_dict.keys():
            if label not in target_dict.keys():     #GTs中没有对应的类别
                self.FP[label] += len(label_dict[label])
                continue
            
            dt_boxes = np.array([obj['box'] for obj in label_dict[label]])
            gt_boxes = np.array(target_dict[label])
            #print(dt_boxes)
            #print(gt_boxes)
            t_TP, t_FP, t_FN = get_TP_FP_FN(dt_boxes, gt_boxes, self.iou_thre)
            #print(label, t_TP, t_FP, t_FN)
            self.TP[label] += t_TP
            self.FP[label] += t_FP
            self.FN[label] += t_FN
    
    def get_precison_recall(self):
        precision = self.TP[1:] / (self.TP[1:] + self.FP[1:])       #第零个标签是背景，不用计算
        recall = self.TP[1:] / (self.TP[1:] + self.FN[1:])
        #print(precision)
        #print(recall)
        return precision, recall

    def get_ap(self, result_list, target_list):
        num = 20
        precison_list = []
        recall_list = []
        for i in range(1,num+1):
            self.TP = np.zeros(self.num_class,dtype=np.int32)
            self.FP = np.zeros(self.num_class,dtype=np.int32)
            self.FN = np.zeros(self.num_class,dtype=np.int32)
            for result, target in zip(result_list, target_list):
                t_result = {k : v.clone().detach()[:i] for k, v in result.items()}  #只要前i个结果
                #print(len(t_result['labels']))
                #t_result['labels'] = t_result['labels'][:i]            
                #t_result['scores'] = t_result['scores'][:i]
                #t_result['boxes']  = t_result['boxes'][:i]
                #print(len(t_result['labels']))
                self.get_label(t_result, target)
            
            p, r = self.get_precison_recall()
            precison_list.append(p)
            recall_list.append(r)
        
        print(precison_list)
        print(recall_list)
        np.save("../result/save_precison3", np.array(precison_list))
        np.save("../result/save_recall3", np.array(recall_list))

import matplotlib.pyplot as plt

def compute_ap(precison_list, recall_list):
    label_num = precison_list.shape[1]
    ap_list = []
    for i in range(label_num):
        precison = precison_list[:,i]
        recall = recall_list[:,i]
        precison = np.insert(precison,0,1)
        recall = np.insert(recall,0,0)
        print(precison)
        print(recall)
        f = 2*precison*recall/(precison+recall)
        print("f1=")
        print(f)
        ap = 0
        last_recall_val = 0
        last_precison_val = 1
        i = 1
        while i < len(recall):
            max_idx = np.argmax(precison[i:])
            max_val = np.max(precison[i:])
            max_idx += i
            #print(max_idx)
            #print(max_val)
            if last_precison_val > max_val:
                ap += ((last_precison_val+max_val)*(recall[max_idx]-last_recall_val))/2.
                last_precison_val = max_val
                last_recall_val = recall[max_idx]
                i = max_idx + 1
            else:
                ap += max_val*(recall[max_idx]-last_recall_val)
                last_precison_val = max_val
                last_recall_val = recall[max_idx]
                i = max_idx + 1
        ap_list.append(ap)
    mAP = sum(ap for ap in ap_list) / label_num
    return ap_list, mAP

classes_name =['background','ore carrier','passenger ship','container ship','bulk cargo carrier','general cargo ship','fishing boat']
def draw_pr():
    precison_list = np.load("../result/save_precison.npy")
    recall_list = np.load("../result/save_recall.npy")
    precison2_list = np.load("../result/save_precison2.npy")
    recall2_list = np.load("../result/save_recall2.npy")
    precison3_list = np.load("../result/save_precison3.npy")
    recall3_list = np.load("../result/save_recall3.npy")
    #print(precison_list)
    #print(recall_list)
    label_num = precison_list.shape[1]
    #print(precison_list[:,0])

    ap1_list, mAP1 = compute_ap(precison_list, recall_list)
    ap2_list, mAP2 = compute_ap(precison2_list, recall2_list)
    ap3_list, mAP3 = compute_ap(precison3_list, recall3_list)
    print(ap1_list, mAP1)
    print(ap2_list, mAP2)
    print(ap3_list, mAP3)
    
    plt.rcParams['figure.figsize'] = (10.0, 7.5)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    for i in range(label_num):
        precison = precison_list[:,i]
        recall = recall_list[:,i]
        precison = np.insert(precison,0,1)
        recall = np.insert(recall,0,0)

        precison2 = precison2_list[:,i]
        recall2 = recall2_list[:,i]
        precison2 = np.insert(precison2,0,1)
        recall2 = np.insert(recall2,0,0)

        precison3 = precison3_list[:,i]
        recall3 = recall3_list[:,i]
        precison3 = np.insert(precison3,0,1)
        recall3 = np.insert(recall3,0,0)
        
        f = 2*precison*recall/(precison+recall)

        plt.subplot(231+i)
        plt.grid()
        plt.title(classes_name[i+1])
        plt.plot(recall, precison, color='red', linewidth=2.0, label="epoch5-lr005")
        plt.plot(recall2, precison2, color='green', linewidth=2.0, label="epoch10-lr005")
        plt.plot(recall2, precison2, color='yellow', linewidth=2.0, label="epoch5-lr002")
        plt.xlabel("recall")
        plt.ylabel("precison")
        plt.legend(["epoch5-lr005","epoch10-lr005","epoch5-lr002"], loc="best")
    plt_name = "pr_plot.jpg"
    plt.savefig(plt_name)
    



def test_one(idx):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 7)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 7)

    model.load_state_dict(torch.load("../model/mask_rcnn_5_2_002.pth"))
    model.to(device)
    model.eval()

    data = SeashipDataset("../../SeaShips", None)
    evaluator = Evaluator(7,0.5)
    targets = []
    image, target = data.__getitem__(idx)
    img_var = [F.to_tensor(image).to(device)]
    original_image_sizes = [img.shape[-2:] for img in img_var]
    targets.append(target)
    targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
    result = model.forward(img_var)
    #print(target['labels'])
    #print(result)
    evaluator.get_label(result[0], target)
    boxes = result[0]['boxes'].cpu().detach().numpy()
    scores = result[0]['scores'].cpu().detach().numpy()
    masks = result[0]['masks'].cpu()

    index = np.where(scores > 0.5)
    boxes = boxes[index]
    masks = masks[index]
    masks= torch.where(masks > 0.5, torch.full_like(masks, 1), torch.full_like(masks, 0))
    m = torch.zeros(original_image_sizes[0])
    for mask in masks:
        m += mask[0]
    m = torch.where(m > 0.5, torch.full_like(m, 1), torch.full_like(m, 0))

    image = image.convert("RGBA")
    mask = TensorToPIL(m)
    mask.convert("RGBA")
    drawRectangle(image, boxes)
    image = drawMasks(image, mask)

    res_name = "../result/test{:0>6d}.png".format(idx)
    image.save(res_name)

def test():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 7)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 7)

    model.load_state_dict(torch.load("../model/mask_rcnn_5_2_002.pth"))
    model.to(device)
    model.eval()

    data = SeashipDataset("../../SeaShips", None)
    evaluator = Evaluator(7,0.8)

    with open("./test_data", "r") as f:
        lines = f.readlines()
        test_list = [int(line) for line in lines]
    print(len(test_list))

    result_list = []
    target_list = []
    batch_size = 2
    for idx in range(0,len(test_list),batch_size):
        imgs = []
        targets = []
        try:
            for i in range(idx, idx+batch_size):
                img, target = data.__getitem__(test_list[i]-1)
                imgs.append(F.to_tensor(img).to(device))
                target = {k : v.to(device) for k, v in target.items()}
                targets.append(target)

            results = model.forward(imgs)

            #result_list.extend(results)
            #target_list.extend(targets)
            for result, target in zip(results, targets):
                target = {k : v.cpu().detach() for k, v in target.items()}
                result = {k : v.cpu().detach() for k, v in result.items()}
                result_list.append(result)
                target_list.append(target)
                #evaluator.get_label(result, target)
        except:
            print(str(traceback.format_exc()))
        
        if idx % 12 == 0:
            print(idx)
    
    evaluator.get_ap(result_list, target_list)

import random

def rand_data():
    idx_list = [i for i in range(1,16001)]
    random.shuffle(idx_list)
    #print(idx_list)

    train_list = idx_list[:14000]
    test_list = idx_list[14000:]
    
    with open("./train_data","w+") as f:
        w_list = [str(id)+'\n' for id in train_list]
        f.writelines(w_list)
    with open("./test_data","w+") as f:
        w_list = [str(id)+'\n' for id in test_list]
        f.writelines(w_list)

if __name__ == "__main__":
    #test(15666)
    #test()
    draw_pr()
    #test_one(523)
    #rand_data()
    #print(torch.cuda.get_device_name(0))
    #print(torch.cuda.device_count())