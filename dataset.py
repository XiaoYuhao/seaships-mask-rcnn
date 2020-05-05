import os 
import numpy as np 
import torch
from PIL import Image

from xml.dom.minidom import parse
import xml.dom.minidom

class SeashipDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.annotation = list(sorted(os.listdir(os.path.join(root,"Annotations"))))
        self.masks = list(sorted(os.listdir(os.path.join(root,"MaskImages"))))
        self.labels ={
            'background' : 0,
            'ore carrier' : 1,
            'passenger ship' : 2,
            'container ship' : 3,
            'bulk cargo carrier' : 4,
            'general cargo ship' : 5,
            'fishing boat' : 6
        } 
        self.classes =['background','ore carrier','passenger ship','container ship','bulk cargo carrier','general cargo ship','fishing boat']
    
    def readxml(self, path):
        DOMTree = xml.dom.minidom.parse(path)
        annotaion = DOMTree.getElementsByTagName('annotation')
        objlist = annotaion[0].getElementsByTagName('object')
        boxes = []
        labels = []
        for obj in objlist:
            name = obj.getElementsByTagName('name')[0].firstChild.data
            #print(name)
            labels.append(self.labels[name])
            box = obj.getElementsByTagName('bndbox')[0]
            x1 = int(box.getElementsByTagName('xmin')[0].firstChild.data)
            y1 = int(box.getElementsByTagName('ymin')[0].firstChild.data)
            x2 = int(box.getElementsByTagName('xmax')[0].firstChild.data)
            y2 = int(box.getElementsByTagName('ymax')[0].firstChild.data)
            #print(x1,y1,x2,y2)
            boxes.append([x1,y1,x2,y2])
        return labels,boxes
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        anno_path = os.path.join(self.root, "Annotations", self.annotation[idx])
        mask_path = os.path.join(self.root, "MaskImages",self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        labels, boxes = self.readxml(anno_path)
        num_objs = len(labels)
        mask = Image.open(mask_path)
        mask = np.array(mask)
        #masks = mask == 255
        masks = np.zeros([num_objs,mask.shape[0],mask.shape[1]],dtype=np.int8)
        #print(np.sum(masks == True))
        #print(labels)
        #print(boxes)
        idx = 0
        for box in boxes:
            obj_mask = np.zeros([mask.shape[0],mask.shape[1]],dtype=np.int16)
            obj_mask[box[1]:box[3],box[0]:box[2]] = mask[box[1]:box[3],box[0]:box[2]]
            obj_mask = obj_mask > 0             #大于0即为标记的mask
            masks[idx,:,:] = obj_mask
            idx += 1
        
        masks = masks == 1
        #print(masks)
        #print(np.sum(masks == True))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target= {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def getitem2(self, idx, epoch):
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        anno_path = os.path.join(self.root, "Annotations", self.annotation[idx])
        mask_dir = "TempMasks{:d}".format(epoch)
        mask_path = os.path.join(self.root, mask_dir ,self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        labels, boxes = self.readxml(anno_path)
        num_objs = len(labels)
        mask = Image.open(mask_path)
        mask = np.array(mask)
        #masks = mask == 255
        masks = np.zeros([num_objs,mask.shape[0],mask.shape[1]],dtype=np.int8)
        #print(np.sum(masks == True))
        #print(labels)
        #print(boxes)
        idx = 0
        for box in boxes:
            obj_mask = np.zeros([mask.shape[0],mask.shape[1]],dtype=np.int16)
            obj_mask[box[1]:box[3],box[0]:box[2]] = mask[box[1]:box[3],box[0]:box[2]]
            obj_mask = obj_mask == 255
            masks[idx,:,:] = obj_mask
            idx += 1
        
        masks = masks == 1
        #print(masks)
        #print(np.sum(masks == True))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target= {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def updatemask(self, idx, mask, epoch):
        old_mask_dir = "TempMasks{:d}".format(epoch)      
        new_mask_dir = "TempMasks{:d}".format(epoch+1)       #第i个tempmask文件夹保存的是i-1轮的结果
        old_mask_path = os.path.join(self.root, old_mask_dir, self.masks[idx])
        new_mask_path = os.path.join(self.root, new_mask_dir, self.masks[idx])
        old_mask = Image.open(old_mask_path)
        np_old_mask = np.array(old_mask)
        #old_mask = np.array(Image.open(mask_path))
        old_num = np.sum(np_old_mask)
        new_mask = np.array(mask)
        num = np.sum(new_mask)
        if num < old_num*0.5 :
            old_mask.save(new_mask_path)
            return False
        mask.save(new_mask_path)
        print(new_mask_path)
        return True


    def __len__(self):
        return len(self.imgs)


def total():
    name_dict = dict()
    data = SeashipDataset("../../SeaShips", None)

    for idx in range(16000):
        path = os.path.join(data.root, "Annotations", data.annotation[idx])
        labels, boxes = data.readxml(path)
        for label in labels:
            if label not in name_dict.keys():
                name_dict[label] = 0
            name_dict[label] += 1
        if idx % 100 == 0:
            print(idx)

    print(name_dict)
    total = sum(num for num in name_dict.values())
    print(total)

'''
{   'ore carrier': 2199, 
    'passenger ship': 474, 
    'container ship': 901, 
    'bulk cargo carrier': 1952, 
    'general cargo ship': 1505, 
    'fishing boat': 2190    }
total: 9221

after data augmentation
{   'ore carrier': 5007, 
    'passenger ship': 1077, 
    'container ship': 2052, 
    'bulk cargo carrier': 4457, 
    'general cargo ship': 3444, 
    'fishing boat': 5037    }
total: 21074
'''

if __name__ == '__main__':
    #S = SeashipDataset('..',None)
    #S.__getitem__(15)
    total()


'''
一个值得注意的点：第一次统计数据的时候很慢，而过一会在统计第二次
则飞快。应该是刚开始从磁盘读取大量数据需要花费很大时间，而之后可
从缓存中读取数据，则会很快。
'''