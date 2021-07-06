import os
import numpy as np
# Note:
# Use itk here will cause deadlock after the first training epoch 
# when using multithread (dataloader num_workers > 0) but reason unknown
import SimpleITK as sitk
from pandas import DataFrame
from utils import read_image

def keep_largest_component(image, largest_n=1):
    c_filter = sitk.ConnectedComponentImageFilter()
    obj_arr = sitk.GetArrayFromImage(c_filter.Execute(image))
    obj_num = c_filter.GetObjectCount()

    obj_vol = np.zeros(obj_num, dtype=np.int64)
    for obj_id in range(obj_num):
        tmp_arr = np.zeros_like(obj_arr)
        tmp_arr[obj_arr == obj_id+1] = 1
        obj_vol[obj_id] = np.sum(tmp_arr)

    sorted_obj_id = np.argsort(obj_vol)[::-1]
    
    tmp_arr = np.zeros_like(obj_arr)
    for i in range(largest_n):
        tmp_arr[obj_arr == sorted_obj_id[i]+1] = 1
    output = sitk.GetImageFromArray(tmp_arr)
    output.SetSpacing(image.GetSpacing())
    output.SetOrigin(image.GetOrigin())
    output.SetDirection(image.GetDirection())

    return output

def cal_dsc(pd, gt):
    y = (np.sum(pd * gt) * 2 + 1) / (np.sum(pd * pd + gt * gt) + 1)
    return y

def cal_asd(a, b):
    filter1 = sitk.SignedMaurerDistanceMapImageFilter()
    filter1.SetUseImageSpacing(True)
    filter1.SetSquaredDistance(False)
    a_dist = filter1.Execute(a)

    a_dist = sitk.GetArrayFromImage(a_dist)
    a_dist = np.abs(a_dist)
    a_edge = np.zeros(a_dist.shape, a_dist.dtype)
    a_edge[a_dist == 0] = 1
    a_num = np.sum(a_edge)

    filter2 = sitk.SignedMaurerDistanceMapImageFilter()
    filter2.SetUseImageSpacing(True)
    filter2.SetSquaredDistance(False)
    b_dist = filter2.Execute(b)

    b_dist = sitk.GetArrayFromImage(b_dist)
    b_dist = np.abs(b_dist)
    b_edge = np.zeros(b_dist.shape, b_dist.dtype)
    b_edge[b_dist == 0] = 1
    b_num = np.sum(b_edge)

    a_dist[b_edge == 0] = 0.0
    b_dist[a_edge == 0] = 0.0

    #a2b_mean_dist = np.sum(b_dist) / a_num
    #b2a_mean_dist = np.sum(a_dist) / b_num
    asd = (np.sum(a_dist) + np.sum(b_dist)) / (a_num + b_num)

    return asd

def eval(pd_path, gt_entries, label_map, cls_num, metric_fn, calc_asd=True, keep_largest=False):
    results = []
    result_lines = ''
    print_line = '\n --- Start calculating metrics --- '
    print(print_line)
    result_lines += '{}\n'.format(print_line)
    for [d_name, casename, _, __, gt_fname] in gt_entries:
        gt_label = read_image(fname=gt_fname)
        gt_array = sitk.GetArrayFromImage(gt_label)
        gt_array = gt_array.astype(dtype=np.uint8)

        # map labels
        tmp_array = np.zeros_like(gt_array)
        lmap = label_map[d_name]
        tgt_labels = []
        for key in lmap:
            tmp_array[gt_array == key] = lmap[key]
            if lmap[key] not in tgt_labels:
                tgt_labels.append(lmap[key])
        gt_array = tmp_array

        for c in tgt_labels:
            pd_fname = '{}/{}@{}@{}.nii.gz'.format(pd_path, d_name, casename, c)
            pd_im = read_image(fname=pd_fname)
            pd_im.SetSpacing(gt_label.GetSpacing())
            pd_im.SetOrigin(gt_label.GetOrigin())
            pd_im.SetDirection(gt_label.GetDirection())
            if keep_largest:
                pd_im = keep_largest_component(pd_im, largest_n=2)
            pd = sitk.GetArrayFromImage(pd_im)
            pd = pd.astype(dtype=np.uint8)
            pd = np.reshape(pd, -1)

            gt = np.zeros_like(gt_array)
            gt[gt_array == c] = 1
            gt_im = sitk.GetImageFromArray(gt)
            gt_im.SetSpacing(gt_label.GetSpacing())
            gt_im.SetOrigin(gt_label.GetOrigin())
            gt_im.SetDirection(gt_label.GetDirection())
            gt = np.reshape(gt, -1)

            dsc = cal_dsc(pd, gt)
            if calc_asd:
                asd = cal_asd(pd_im, gt_im)
            else:
                asd = 0
            results.append([d_name, casename, c, dsc, asd])

            print_line = ' --- {0:s}@{1:s}@{2:d}:\t\tDSC = {3:.2f}%\tASD = {4:.2f}mm'.format(d_name, casename, c, dsc*100.0, asd)
            print(print_line)
            result_lines += '{}\n'.format(print_line)
    
    df = DataFrame(results, columns=['Dataset', 'Case', 'Class', 'DSC', 'ASD'])
    df.to_csv('{}/{}.csv'.format(pd_path, metric_fn))

    dsc = []
    asd = []
    dsc_mean = 0
    asd_mean = 0
    for c in range(cls_num):
        dsc_m = df[df['Class'] == c+1]['DSC'].mean()
        dsc_v = df[df['Class'] == c+1]['DSC'].std()
        asd_m = df[df['Class'] == c+1]['ASD'].mean()
        asd_v = df[df['Class'] == c+1]['ASD'].std()
        dsc.append([dsc_m, dsc_v])        
        asd.append([asd_m, asd_v])
        dsc_mean += dsc_m
        asd_mean += asd_m
        print_line = ' --- class {0:d}:\tDSC = {1:.2f}({2:.2f})%\tASD = {3:.2f}({4:.2f})mm'.format(c+1, dsc_m*100.0, dsc_v*100.0, asd_m, asd_v)
        print(print_line)
        result_lines += '{}\n'.format(print_line)
    dsc_mean = dsc_mean / cls_num
    asd_mean = asd_mean / cls_num
    dsc = np.array(dsc)
    asd = np.array(asd)

    print_line = ' --- class-avg:\tDSC = {0:.2f}%\tASD = {1:.2f}mm'.format(dsc_mean*100.0, asd_mean)
    print(print_line)
    result_lines += '{}\n'.format(print_line)
    print_line = ' --- Finish calculating metrics --- \n'
    print(print_line)
    result_lines += '{}\n'.format(print_line)

    result_fn = '{}/{}.txt'.format(pd_path, metric_fn)
    with open(result_fn, 'w') as result_file:
        result_file.write(result_lines)
    
    return dsc, asd, dsc_mean, asd_mean
