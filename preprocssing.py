from pathlib import Path
import nibabel as nib
import numpy as np
from glob import glob

inp_images = glob("LGG-20220401T063338Z-001/LGG/*/*flair.nii")
seg_images = []
for i in inp_images:
    seg_images.append(i.replace("flair.nii", "seg.nii"))


def normalize(data):
    """
    Normalize the data to the range 0-1
    """
    normalized_data = (data - data.min()) / (data.max() - data.min())
    return normalized_data


def bounding_box(img):
    """
    removes all the rows with all 0.0(s) and retuns the coordinates of remaining part of image

    returns [x1, y1, x2, y2]
    """
    y1 = -1
    for each_row in img:
        if((each_row==np.array([0.0 for i in range(160)])).all()): y1 += 1
        else: break

    y2 = len(img)
    for each_row_ind in range(len(img)):
        if((img[-(each_row_ind+1)] == np.array([0.0 for i in range(160)])).all()): y2 -= 1
        else: break

    x1 = -1
    transpose_lab = img.T
    for each_col in transpose_lab:
        if((each_col == np.array([0.0 for i in range(160)])).all()): x1 += 1
        else: break

    x2 = len(img)
    transpose_lab = img.T
    for each_row_ind in range(len(img)):
        if((transpose_lab[-(each_row_ind+1)] == np.array([0.0 for i in range(160)])).all()): x2 -= 1
        else: break

    if(x1>=x2 or y1>=y2): return [0, 0, 0, 0]
    
    return [x1, y1, x2, y2]


def find_area(coords):
    x1 = coords[0]
    y1 = coords[1]
    x2 = coords[2]
    y2 = coords[3]

    length = x2-x1
    height = y2-y1

    return length*height


def best_img_indices(segment_img):
    max_area = 0
    final_mask_ind = -1
    indices = []

    for i in range(segment_img.shape[2]):
        cur_area = find_area(bounding_box(segment_img[:,:,i]))
        if(cur_area>max_area):
            max_area = cur_area
            final_mask_ind = i
    
    for ind in range(final_mask_ind-10, final_mask_ind+11):
        if(ind<0 or ind>=segment_img.shape[0]):
            continue
        
        if(find_area(bounding_box(segment_img[:,:,ind])) > max_area*0.75):
            indices.append(ind)
    
    return indices


folder = Path("new_prepro")
counter = 0

for img, seg in zip(inp_images, seg_images):

    loaded_img = nib.load(img)
    loaded_seg = nib.load(seg)

    cropped_img = np.array(loaded_img.get_fdata()[40:-40, 40:-40])
    cropped_seg = np.array(loaded_seg.get_fdata()[40:-40, 40:-40])

    normalized_img = normalize(cropped_img)
    normalized_seg = normalize(cropped_seg)

    best_indices = best_img_indices(normalized_seg)

    cur_folder = folder/str(counter)
    img_folder = cur_folder/"data"
    seg_folder = cur_folder/"mask"

    img_folder.mkdir(parents=True, exist_ok=True)
    seg_folder.mkdir(parents=True, exist_ok=True)

    for i in best_indices:
        np.save(img_folder/str(i), np.array(normalized_img[:, :, i]))
        np.save(seg_folder/str(i), np.array(normalized_seg[:, :, i]))
    
    counter += 1