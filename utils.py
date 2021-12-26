import nibabel as nib
import numpy as np
import os

def get_file_list(path):
    allfile = os.listdir(path)
    allnii = [i for i in allfile if 'nii.gz' in i]
    ct = [i for i in allnii if 'seg' not in i]
    mask = [i for i in allnii if 'seg' in i]
    assert len(ct)==len(mask)
    return ct, mask

# 若方向不同
def redirect(img):

    # original_affine = img.affine
    original_axcode = nib.aff2axcodes(img.affine)
    # print(nii.split('/')[-1], original_axcode)
    img = img.as_reoriented(nib.io_orientation(img.affine))
    new_axcode = nib.aff2axcodes(img.affine)
    print('original axcode', original_axcode, 'now (should be ras)', new_axcode)
    return img
    # nib.save(img, image)
    # save_pickle((original_affine, original_axcode), origaffine_pkl)

def Hu2Gray(ctshot):
    dfactor = 255.0 / (np.max(ctshot) - np.min(ctshot))
    ctshot = (ctshot - np.min(ctshot)) * dfactor
    ctshot = ctshot.astype(np.uint8)
    return ctshot

