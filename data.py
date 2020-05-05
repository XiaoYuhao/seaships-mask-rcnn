import cv2
import numpy as np
from dataset import SeashipDataset
import random, os
from xml.dom.minidom import parse
import xml.dom.minidom

def rotate_im(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    image = cv2.warpAffine(image, M, (nW, nH))
    
    return image

def get_corners(boxes):
    '''
    Arg： 
        numpy boxes
    Return:
        numpy corners
        边界框四个角的坐标
    '''
    width = (boxes[:, 2] - boxes[:, 0]).reshape(-1, 1)
    height = (boxes[:, 3] - boxes[:, 1]).reshape(-1, 1)

    x1 = boxes[:, 0].reshape(-1, 1)
    y1 = boxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = boxes[:, 2].reshape(-1,1)
    y4 = boxes[:, 3].reshape(-1,1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners

def rotate_box(corners, angle, cx, cy, h, w):
    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    calculated = np.dot(M, corners.T).T
    calculated = calculated.reshape(-1, 8)

    return calculated

def get_enclosing_box(corners):
    x_ = corners[:, [0,2,4,6]]
    y_ = corners[:, [1,3,5,7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[: 8:]))

    return final

def DrawRect(img, boxes):
    for box in boxes:
        w = box[2] - box[0]
        h = box[3] - box[1]
        cv2.rectangle(img, (box[0], box[1]), (box[0]+w, box[1]+h), (0,255,0), 2)
    return img

def rotate(img, boxes, angle):
    #angle = 15
    w, h = img.shape[1], img.shape[0]
    cx, cy = w//2, h//2
    img = rotate_im(img, angle)

    corners = get_corners(boxes)
    corners = np.hstack((corners, boxes[:, 4:]))
    corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h ,w)
    new_boxes = get_enclosing_box(corners)

    scale_factor_x = img.shape[1] / w
    scale_factor_y = img.shape[0] / h

    img = cv2.resize(img, (w,h))
    new_boxes[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]
    boxes = new_boxes
    #print(boxes)
    #boxes = clip_box(boxes, [0,0,w,h], 0.25)
    #img = DrawRect(img, boxes)

    return img, boxes

def add_gaussian_noise(image_in, noise_sigma):
    temp_image = np.float64(np.copy(image_in))

    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.rand(h, w) * noise_sigma

    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:,:,0] = temp_image[:,:,0] + noise
        noisy_image[:,:,1] = temp_image[:,:,1] + noise
        noisy_image[:,:,2] = temp_image[:,:,2] + noise

    return noisy_image

def horizontal_flip(img, boxes):
    w, h = img.shape[1], img.shape[0]
    img = cv2.flip(img, 1)
    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    return img, boxes

def change_bright(img, lower=0.5, upper=1.25):
    mean = np.mean(img)
    img = img - mean
    img = img * random.uniform(lower, upper) + mean * random.uniform(lower, upper)
    #img = img / 255.
    return img

def readxml(path):
    DOMTree = xml.dom.minidom.parse(path)
    annotaion = DOMTree.getElementsByTagName('annotation')
    objlist = annotaion[0].getElementsByTagName('object')
    boxes = []
    for obj in objlist:
        box = obj.getElementsByTagName('bndbox')[0]
        x1 = int(box.getElementsByTagName('xmin')[0].firstChild.data)
        y1 = int(box.getElementsByTagName('ymin')[0].firstChild.data)
        x2 = int(box.getElementsByTagName('xmax')[0].firstChild.data)
        y2 = int(box.getElementsByTagName('ymax')[0].firstChild.data)
        #print(x1,y1,x2,y2)
        boxes.append([x1,y1,x2,y2])
    boxes = np.array(boxes, dtype=np.float)
    return boxes

def writeXml(src_path, des_path, filename, boxes):
    DOMTree = xml.dom.minidom.parse(src_path)
    annotaion = DOMTree.getElementsByTagName('annotation')
    name = annotaion[0].getElementsByTagName('filename')[0]
    name.firstChild.data = filename
    path = annotaion[0].getElementsByTagName('path')[0]
    real_path = "SeaShips/JPEGImages/" + filename
    path.firstChild.data = real_path
    objlist = annotaion[0].getElementsByTagName('object')
    for i, obj in enumerate(objlist):
        box = obj.getElementsByTagName('bndbox')[0]
        box.getElementsByTagName('xmin')[0].firstChild.data = int(boxes[i][0])
        box.getElementsByTagName('ymin')[0].firstChild.data = int(boxes[i][1])
        box.getElementsByTagName('xmax')[0].firstChild.data = int(boxes[i][2])
        box.getElementsByTagName('ymax')[0].firstChild.data = int(boxes[i][3])
    
    with open(des_path, 'w') as f:
        f.write(DOMTree.toprettyxml(indent='\t'))



def Data_Augmentation():
    src_img_num = 7000
    idx_list = [i for i in range(1,src_img_num+1)]
    #print(idx_list)
    random.shuffle(idx_list)
    extend_list = idx_list[:2000]           #额外再加1000张
    idx_list.extend(extend_list)
    num = 0
    for idx in idx_list:
        src_img_name = "{:0>6d}.jpg".format(idx)
        src_img_path = os.path.join("../../SeaShips/JPEGImages/", src_img_name)
        src_anno_name = "{:0>6d}.xml".format(idx)
        src_anno_path = os.path.join("../../SeaShips/Annotations/", src_anno_name)
        src_mask_name = "{:0>6d}.png".format(idx)
        src_mask_path = os.path.join("../../SeaShips/MaskImages", src_mask_name)
        #print(src_mask_path)
        des_img_name = "{:0>6d}.jpg".format(num+7001)
        des_img_path = os.path.join("../../SeaShips/JPEGImages/", des_img_name)
        des_anno_name = "{:0>6d}.xml".format(num+7001)
        des_anno_path = os.path.join("../../SeaShips/Annotations/", des_anno_name)
        des_mask_name = "{:0>6d}.png".format(num+7001)
        des_mask_path = os.path.join("../../SeaShips/MaskImages", des_mask_name)

        if num < 1000:      #旋转15度
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
            temp = np.copy(boxes)
            img, boxes = rotate(img, boxes, 15)
            mask, temp = rotate(mask, temp, 15) 
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)
        
        elif num < 2000:    #旋转-15度
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
            temp = np.copy(boxes)
            img, boxes = rotate(img, boxes, -15)
            mask, temp = rotate(mask, temp, -15) 
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)

        elif num < 3000:    #水平翻转
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
            temp = np.copy(boxes)
            img, boxes = horizontal_flip(img, boxes)
            mask, temp = horizontal_flip(mask, temp)
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)
        
        elif num < 4000:    #添加高斯噪声
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)

            noise_sigma = 60
            img = add_gaussian_noise(img, noise_sigma)
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cmd = ("cp %s %s" %(src_mask_path, des_mask_path))
            os.popen(cmd)

        elif num < 5000:    #随机亮度和对比度
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)

            img = change_bright(img)
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cmd = ("cp %s %s" %(src_mask_path, des_mask_path))
            os.popen(cmd)
        
        elif num < 5500:    #旋转15度并加高斯模糊
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
            temp = np.copy(boxes)
            noise_sigma = 30
            img = add_gaussian_noise(img, noise_sigma)
            img, boxes = rotate(img, boxes, 15)
            mask, temp = rotate(mask, temp, 15) 
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)
    
        elif num < 6000:    #旋转-15度并加高斯模糊
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
            temp = np.copy(boxes)
            noise_sigma = 30
            img = add_gaussian_noise(img, noise_sigma)
            img, boxes = rotate(img, boxes, -15)
            mask, temp = rotate(mask, temp, -15) 
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)

        elif num < 6500:    #旋转15度并加随机亮度
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
            temp = np.copy(boxes)
            img = change_bright(img)
            img, boxes = rotate(img, boxes, 15)
            mask, temp = rotate(mask, temp, 15) 
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)
    
        elif num < 7000:    #旋转-15度并加随机亮度
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
            temp = np.copy(boxes)
            img = change_bright(img)
            img, boxes = rotate(img, boxes, -15)
            mask, temp = rotate(mask, temp, -15) 
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)

        elif num < 7500:    #水平翻转加随机亮度
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
            temp = np.copy(boxes)
            img, boxes = horizontal_flip(img, boxes)
            mask, temp = horizontal_flip(mask, temp)
            img = change_bright(img)
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)
        
        elif num < 8000:    #水平翻转加高斯模糊
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
            temp = np.copy(boxes)
            img, boxes = horizontal_flip(img, boxes)
            mask, temp = horizontal_flip(mask, temp)
            noise_sigma = 25
            img = add_gaussian_noise(img, noise_sigma)
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)
        
        elif num < 8500:    #旋转15度并加高斯噪声、随机亮度
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
            temp = np.copy(boxes)
            noise_sigma = 20
            img = add_gaussian_noise(img, noise_sigma)
            img = change_bright(img)
            img, boxes = rotate(img, boxes, 10)
            mask, temp = rotate(mask, temp, 10) 
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)
        
        else:
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
            temp = np.copy(boxes)
            noise_sigma = 20
            img = add_gaussian_noise(img, noise_sigma)
            img = change_bright(img)
            img, boxes = rotate(img, boxes, -10)
            mask, temp = rotate(mask, temp, -10) 
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)
        
        if num % 100 == 0:
            print(num)
        num += 1



def test():
    img_list = os.listdir("../SeaShips/JPEGImages")
    img_num = len(img_list)
    idx_list = [i for i in range(img_num)]
    #random.shuffle(idx_list)
    #extend_list = idx_list[:2000]           #额外再加1000张
    #idx_list.extend(extend_list)
    num = 0
    
    for idx in range(1, 16):
        src_img_name = "{:0>6d}.jpg".format(idx)
        src_img_path = os.path.join("../../SeaShips/JPEGImages/", src_img_name)
        src_anno_name = "{:0>6d}.xml".format(idx)
        src_anno_path = os.path.join("../../SeaShips/Annotations/", src_anno_name)
        src_mask_name = "{:0>6d}.png".format(idx)
        src_mask_path = os.path.join("../../SeaShips/MaskImages", src_mask_name)

        des_img_name = "{:0>6d}.jpg".format(idx+7001)
        des_img_path = os.path.join("../../SeaShips/JPEGImages/", des_img_name)
        des_anno_name = "{:0>6d}.xml".format(idx+7001)
        des_anno_path = os.path.join("../../SeaShips/Annotations/", des_anno_name)
        des_mask_name = "{:0>6d}.png".format(idx+7001)
        des_mask_path = os.path.join("../../SeaShips/MaskImages", des_mask_name)

        if num < 1:      #旋转15度
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path)
            temp = np.copy(boxes)
            img, boxes = rotate(img, boxes, 15)
            mask, temp = rotate(mask, temp, 15) 
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)
        
        elif num < 2:    #旋转-15度
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path)
            temp = np.copy(boxes)
            img, boxes = rotate(img, boxes, -15)
            mask, temp = rotate(mask, temp, -15) 
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)

        elif num < 3:    #水平翻转
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path)
            temp = np.copy(boxes)
            img, boxes = horizontal_flip(img, boxes)
            mask, temp = horizontal_flip(mask, temp)
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)
        
        elif num < 4:    #添加高斯噪声
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)

            noise_sigma = 60
            img = add_gaussian_noise(img, noise_sigma)
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cmd = ("cp %s %s" %(src_mask_path, des_mask_path))
            os.popen(cmd)

        elif num < 5:    #随机亮度和对比度
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)

            img = change_bright(img)
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cmd = ("cp %s %s" %(src_mask_path, des_mask_path))
            os.popen(cmd)
        
        elif num < 6:    #旋转15度并加高斯模糊
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path)
            temp = np.copy(boxes)
            noise_sigma = 30
            img = add_gaussian_noise(img, noise_sigma)
            img, boxes = rotate(img, boxes, 15)
            mask, temp = rotate(mask, temp, 15) 
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)
    
        elif num < 7:    #旋转-15度并加高斯模糊
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path)
            temp = np.copy(boxes)
            noise_sigma = 30
            img = add_gaussian_noise(img, noise_sigma)
            img, boxes = rotate(img, boxes, -15)
            mask, temp = rotate(mask, temp, -15) 
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)

        elif num < 8:    #旋转15度并加随机亮度
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path)
            temp = np.copy(boxes)
            img = change_bright(img)
            img, boxes = rotate(img, boxes, 15)
            mask, temp = rotate(mask, temp, 15) 
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)
    
        elif num < 9:    #旋转-15度并加随机亮度
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path)
            temp = np.copy(boxes)
            img = change_bright(img)
            img, boxes = rotate(img, boxes, -15)
            mask, temp = rotate(mask, temp, -15) 
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)

        elif num < 10:    #水平翻转加随机亮度
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path)
            temp = np.copy(boxes)
            img, boxes = horizontal_flip(img, boxes)
            mask, temp = horizontal_flip(mask, temp)
            img = change_bright(img)
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)
        
        elif num < 11:    #水平翻转加高斯模糊
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path)
            temp = np.copy(boxes)
            img, boxes = horizontal_flip(img, boxes)
            mask, temp = horizontal_flip(mask, temp)
            noise_sigma = 25
            img = add_gaussian_noise(img, noise_sigma)
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)
        
        elif num < 12:    #旋转15度并加高斯噪声、随机亮度
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path)
            temp = np.copy(boxes)
            noise_sigma = 20
            img = add_gaussian_noise(img, noise_sigma)
            img = change_bright(img)
            img, boxes = rotate(img, boxes, 10)
            mask, temp = rotate(mask, temp, 10) 
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)
        
        else:
            img = cv2.imread(src_img_path)
            boxes = readxml(src_anno_path)
            mask = cv2.imread(src_mask_path)
            temp = np.copy(boxes)
            noise_sigma = 20
            img = add_gaussian_noise(img, noise_sigma)
            img = change_bright(img)
            img, boxes = rotate(img, boxes, -10)
            mask, temp = rotate(mask, temp, -10) 
            writeXml(src_anno_path, des_anno_path, src_img_name, boxes)
            cv2.imwrite(des_img_path, img)
            cv2.imwrite(des_mask_path, mask)
        
        print(num)
        num += 1

def test_anno():
    img_num = 7000
    idx_list = [i for i in range(1,img_num+1)]
    random.shuffle(idx_list)
    num = 20
    
    for idx in idx_list[:200]:
        des_img_name = "{:0>6d}.jpg".format(idx+7001)
        des_img_path = os.path.join("../../SeaShips/JPEGImages/", des_img_name)
        des_anno_name = "{:0>6d}.xml".format(idx+7001)
        des_anno_path = os.path.join("../../SeaShips/Annotations/", des_anno_name)
        des_mask_name = "{:0>6d}.png".format(idx+7001)
        des_mask_path = os.path.join("../../SeaShips/MaskImages", des_mask_name)
        save_img_name = "{:0>6d}.jpg".format(num+7001)
        save_path = os.path.join("../per_data/", save_img_name)

        boxes = readxml(des_anno_path)
        boxes = boxes.astype(np.int32)
        img = cv2.imread(des_img_path)
        img = DrawRect(img, boxes)
        cv2.imwrite(save_path, img)

        num += 1
        print(save_path)



if __name__ == "__main__":
    
    img_path = "../../SeaShips/JPEGImages/001223.jpg"
    save_src_path = "../per_data/001223.jpg"
    save_des_path = "../per_data/001224.jpg"
    anno_path = "../../SeaShips/Annotations/001223.xml"
    data = SeashipDataset("../../SeaShips", None)
    #img, anno = data.getitem2(1222)
    #boxes = anno['boxes']
    boxes = readxml(anno_path)
    print(boxes)
    
    img = cv2.imread(img_path)
    cv2.imwrite(save_src_path, img)

    img, boxes = horizontal_flip(img, boxes)
    #img = DrawRect(img, boxes)
    cv2.imwrite(save_des_path, img)

    '''

    img, boxes = rotate(img, boxes, 15)
    
    #cv2.imwrite(save_path, new_img)

    noise_sigma = 100
    img = add_gaussian_noise(img, noise_sigma)
    #cv2.imwrite(save_path, noisy_img)

    img, boxes = horizontal_flip(img, boxes)
    img = DrawRect(img, boxes)
    cv2.imwrite(save_path, img)

    src_path = "../SeaShips/Annotations/000524.xml"
    des_path = "../SeaShips/Annotations/007001.xml"
    name = "007001.jpg"
    writeXml(src_path, des_path, name, boxes)
    #new_img, boxes = change_bright(img, boxes)
    #cv2.imwrite(save_path, new_img)

    #Data_Augmentation()
    '''

    #test()
    #test_anno()
    #Data_Augmentation()