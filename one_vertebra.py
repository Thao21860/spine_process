import os.path
import nibabel as nib
import numpy as np
import tqdm
from utils import get_file_list, redirect
import SimpleITK as sitk
'''
根据现有mask裁剪单个锥体
'''
def split_mask(mask):
    labels = {}
    for i in range(1,30):
        x = mask == i
        if x.any():
            labels[i] = x.astype(int)
    return labels

def trans_and(CT, mask):
    CT = ct.copy()
    mask = mask > 0

    return (CT * mask)
def crop_ct(img):
    #  定位，确定三个范围[x,y,z]

    CT = img.copy()
    assert len(CT.shape) == 3

    x_range = find_start_end(CT, 'x')
    y_range = find_start_end(CT, 'y')
    z_range = find_start_end(CT, 'z')
    # 根据范围裁剪图像
    CT = CT[x_range[0]:x_range[1], :, :]

    CT = CT[:, y_range[0]:y_range[1], :]
    CT = CT[:, :, z_range[0]:z_range[1]]
    print("crop size {}".format(CT.shape))

    return CT

def find_start_end(CT, dim='x'):
    split_sum = []
    start, end = 0, 0

    if dim == 'x':
        for i in range(CT.shape[0]):
            split_sum.append(CT[i,:,:].sum())
    elif dim == 'y':
        for i in range(CT.shape[1]):
            split_sum.append(CT[:,i,:].sum())
    else :
        for i in range(CT.shape[2]):
            split_sum.append(CT[:,:,i].sum())
    length = len(split_sum)
    for j in range(length):
        if split_sum[j] != 0:
            start = j
            break
    for k in range(length-1, -1, -1):
        if split_sum[k] != 0:
            end = k
            break

    print(start, end)
    return [start, end]


if __name__ == "__main__":
    rootpath = "D:\\dev\\dataset\\01_training"
    log_txt = open('shape.txt', 'a')
    niis , masks = get_file_list(rootpath)
    for i in tqdm.tqdm(range(len(niis))):
        print(niis[i])
        ct = nib.load(os.path.join(rootpath, niis[i]))
        maskpath = os.path.join(rootpath, niis[i][:-7] + '_seg.nii.gz')
        mask = nib.load(maskpath)
        affine = ct.affine
        header = ct.header
        ct = redirect(ct)
        ct = ct.get_fdata()
        mask = redirect(mask)
        mask = mask.get_fdata()
        labels = split_mask(mask)
        # 遍历标签, 进行裁剪
        for key in labels.keys():
            img = trans_and(ct, labels[key])
            img = crop_ct(img)
            name = niis[i][:-7] + '_' + str(key)
            log_txt.write(name + ' ' + str(img.shape) + '\n')
            img = nib.Nifti1Image(img, affine, header)
            nib.save(img, os.path.join('D:\Epan',name))
            # img = sitk.GetImageFromArray(img)
            # sitk.WriteImage(img, name + '.nii.gz')
            break
        break
    log_txt.close()
