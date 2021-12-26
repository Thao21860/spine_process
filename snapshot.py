import nibabel as nib
import numpy as np
import json
import os
import cv2
from utils import get_file_list, redirect, Hu2Gray
'''
  生成ct图像中间区域的snapshot
'''

def GetMidPic(datapath, nii, mask):

    name = nii[:-7]
    print(name)
    niiPath = os.path.join(datapath, nii)
    maskPath = os.path.join(datapath, name + "_seg.nii.gz")
    image = nib.load(niiPath)
    mask = nib.load(maskPath)
    image = redirect(image)
    mask = redirect(mask)

    pixdim = image.header['pixdim']
    print(pixdim[1:4])    # pixdim 1 2 3 x y z
    image = image.get_fdata()
    mask = mask.get_fdata()
    # 抽取中间层，转灰度图像
    mid = int(image.shape[0]/2)
    mid_mask = mask[mid,:,:]
    ctshot = Hu2Gray(image[mid,:,:])
    # resize为实际尺寸
    width, height = ctshot.shape
    size = (width, height) if width > height else (height, width)
    print(size)
    size = (int(size[0]), int(size[1] * ((pixdim[3]+0.05)/pixdim[2])))
    print(size)
    ctshot = cv2.flip(np.rot90(ctshot, 1), 1)
    mid_mask = np.rot90(mid_mask, 1)
    ctshot = cv2.resize(ctshot, size)
    mid_mask = cv2.resize(mid_mask, size, interpolation=cv2.INTER_NEAREST)  # 必须使用INTER_NEAREST
    # print("ct{} mask{}".format(ctshot.shape, mid_mask.shape))
    return ctshot, mid_mask


def getcolor_from_id(id):
    for i in colorlabel:
        if i['ID'] == id:
            return i['Color']

def showMask(ct, mask, name):
    ctshot3 = cv2.cvtColor(ct, cv2.COLOR_GRAY2RGB)
    w, h = mask.shape
    for i in range(w):
        for j in range(h):
            if mask[i, j] != 0:
                ctshot3[i, j,:] = getcolor_from_id(mask[i, j])[0:3]
    cv2.imwrite(name[:-7] + "_snapshot.jpeg", ctshot3)

if __name__ == "__main__":
    path = 'd:/dev/dataset/result'

    with open('labelcolor.json', encoding='utf-8') as f:
        colorlabel = json.load(f)['ColorLabel']
    print(colorlabel[1]['Color']) # dict

    niis, mask = get_file_list(path)
    print(len(niis))
    for i in range(len(niis)):
        ctshot, mid_mask = GetMidPic(path, niis[i], mask[i])
        # 上色
        showMask(ctshot, mid_mask, niis[i])

