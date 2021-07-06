import os
import sys
import torch
from torch.utils import data
import itk
import numpy as np
import random
import SimpleITK as sitk

def read_image(fname, imtype):
    reader = itk.ImageFileReader[imtype].New()
    reader.SetFileName(fname)
    reader.Update()
    image = reader.GetOutput()
    return image

def scan_path(d_name, d_path):
    entries = []
    if d_name == 'KiTS':
        for f in os.listdir(d_path):
            if f.startswith('volume-') and f.endswith('.mha'):
                id = int(f.split('.mha')[0].split('volume-')[1])
                if os.path.isfile('{}/segmentation-{}.mha'.format(d_path, id)):
                    case_name = 'volume-{}'.format(id)
                    image_name = '{}/volume-{}.mha'.format(d_path, id)
                    label_name = '{}/segmentation-{}.mha'.format(d_path, id)
                    entries.append([d_name, case_name, image_name, label_name])
    elif d_name == 'LiTS':
        for case_name in os.listdir(d_path):
            image_name = '{}/{}/imaging.nii.gz'.format(d_path, case_name)
            image_ld_name = '{}/{}/imaging_ld.nii.gz'.format(d_path, case_name)
            label_name = '{}/{}/segmentation.nii.gz'.format(d_path, case_name)
            if os.path.isfile(image_name) and os.path.isfile(label_name) and os.path.isfile(image_ld_name):
                entries.append([d_name, case_name, image_name, image_ld_name, label_name])
    elif d_name == 'BTCV':
        for f in os.listdir(d_path):
            if f.startswith('volume-'):
                id = int(f.split('.nii')[0].split('volume-')[1])
                if os.path.isfile('{}/segmentation-{}.nii.gz'.format(d_path, id)):
                    case_name = 'volume-{}'.format(id)
                    image_name = '{}/volume-{}.nii.gz'.format(d_path, id)
                    label_name = '{}/segmentation-{}.nii.gz'.format(d_path, id)
                    entries.append([d_name, case_name, image_name, label_name])
    elif d_name == 'spleen':
        for f in os.listdir('{}/imagesTr'.format(d_path)):
            if f.startswith('spleen_'):
                id = int(f.split('.nii.gz')[0].split('spleen_')[1])
                if os.path.isfile('{}/labelsTr/{}'.format(d_path, f)):
                    case_name = 'spleen_{}'.format(id)
                    image_name = '{}/imagesTr/{}'.format(d_path, f)
                    label_name = '{}/labelsTr/{}'.format(d_path, f)
                    entries.append([d_name, case_name, image_name, label_name])
    return entries

def create_folds(data_path, fold_num, exclude_case):
    fold_file_name = '{0:s}/CV_{1:d}-fold.txt'.format(sys.path[0], fold_num)
    folds = {}
    if os.path.exists(fold_file_name):
        with open(fold_file_name, 'r') as fold_file:
            strlines = fold_file.readlines()
            for strline in strlines:
                strline = strline.rstrip('\n')
                params = strline.split()
                fold_id = int(params[0])
                if fold_id not in folds:
                    folds[fold_id] = []
                folds[fold_id].append([params[1], params[2], params[3], params[5], params[4]])
    else:
        entries = []
        for [d_name, d_path] in data_path:            
            entries.extend(scan_path(d_name, d_path))
        for e in entries:
            if e[0:2] in exclude_case:
                entries.remove(e)
        case_num = len(entries)
        fold_size = int(case_num / fold_num)
        random.shuffle(entries)
        for fold_id in range(fold_num-1):
            folds[fold_id] = entries[fold_id*fold_size:(fold_id+1)*fold_size]
        folds[fold_num-1] = entries[(fold_num-1)*fold_size:]

        with open(fold_file_name, 'w') as fold_file:
            for fold_id in range(fold_num):
                for [d_name, case_name, image_path, label_path] in folds[fold_id]:
                    fold_file.write('{0:d} {1:s} {2:s} {3:s} {4:s}\n'.format(fold_id, d_name, case_name, image_path, label_path))

    folds_size = [len(x) for x in folds.values()]

    return folds, folds_size

def normalize(x, min, max):
    factor = 1.0 / (max - min)
    x[x < min] = min
    x[x > max] = max
    x = (x - min) * factor
    return x

def generate_transform(identity):
    if identity:
        t = itk.IdentityTransform[itk.D, 3].New()
    else:
        min_rotate = -0.05 # [rad]
        max_rotate = 0.05 # [rad]
        min_offset = -5.0 # [mm]
        max_offset = 5.0 # [mm]
        t = itk.Euler3DTransform[itk.D].New()
        euler_parameters = t.GetParameters()
        euler_parameters = itk.OptimizerParameters[itk.D](t.GetNumberOfParameters())
        euler_parameters[0] = min_rotate + random.random() * (max_rotate - min_rotate) # rotate
        euler_parameters[1] = min_rotate + random.random() * (max_rotate - min_rotate) # rotate
        euler_parameters[2] = min_rotate + random.random() * (max_rotate - min_rotate) # rotate
        euler_parameters[3] = min_offset + random.random() * (max_offset - min_offset) # tranlate
        euler_parameters[4] = min_offset + random.random() * (max_offset - min_offset) # tranlate
        euler_parameters[5] = min_offset + random.random() * (max_offset - min_offset) # tranlate
        t.SetParameters(euler_parameters)
    return t

def resample(image, imtype, size, spacing, origin, transform, linear, dtype):
    o_origin = image.GetOrigin()
    o_spacing = image.GetSpacing()
    o_size = image.GetBufferedRegion().GetSize()
    output = {}
    output['org_size'] = np.array(o_size, dtype=int)
    output['org_spacing'] = np.array(o_spacing, dtype=float)
    output['org_origin'] = np.array(o_origin, dtype=float)
    
    if origin is None: # if no origin point specified, center align the resampled image with the original image
        new_size = np.zeros(3, dtype=int)
        new_spacing = np.zeros(3, dtype=float)
        new_origin = np.zeros(3, dtype=float)
        for i in range(3):
            new_size[i] = size[i]
            if spacing[i] > 0:
                new_spacing[i] = spacing[i]
                new_origin[i] = o_origin[i] + o_size[i]*o_spacing[i]*0.5 - size[i]*spacing[i]*0.5
            else:
                new_spacing[i] = o_size[i] * o_spacing[i] / size[i]
                new_origin[i] = o_origin[i]
    else:
        new_size = np.array(size, dtype=int)
        new_spacing = np.array(spacing, dtype=float)
        new_origin = np.array(origin, dtype=float)

    output['size'] = new_size
    output['spacing'] = new_spacing
    output['origin'] = new_origin

    resampler = itk.ResampleImageFilter[imtype, imtype].New()
    resampler.SetInput(image)
    resampler.SetSize((int(new_size[0]), int(new_size[1]), int(new_size[2])))
    resampler.SetOutputSpacing((float(new_spacing[0]), float(new_spacing[1]), float(new_spacing[2])))
    resampler.SetOutputOrigin((float(new_origin[0]), float(new_origin[1]), float(new_origin[2])))
    resampler.SetTransform(transform)
    if linear:
        resampler.SetInterpolator(itk.LinearInterpolateImageFunction[imtype, itk.D].New())
    else:
        resampler.SetInterpolator(itk.NearestNeighborInterpolateImageFunction[imtype, itk.D].New())
    resampler.SetDefaultPixelValue(int(np.min(itk.GetArrayFromImage(image))))
    resampler.Update()
    rs_image = resampler.GetOutput()
    image_array = itk.GetArrayFromImage(rs_image)
    image_array = image_array[np.newaxis, :].astype(dtype)
    output['array'] = image_array

    return output

def make_onehot(input, cls):
    oh = np.repeat(np.zeros_like(input), cls*2, axis=0)
    for i in range(cls):
        tmp = np.zeros_like(input)
        tmp[input==i+1] = 1
        oh[i*2+0,:] = 1-tmp
        oh[i*2+1,:] = tmp
    return oh

def make_flag(cls, labelmap):
    flag = np.zeros([cls, 1])
    for key in labelmap:
        flag[labelmap[key]-1,0] = 1
    return flag

def image2file(image, imtype, fname):
    writer = itk.ImageFileWriter[imtype].New()
    writer.SetInput(image)
    writer.SetFileName(fname)
    writer.Update()

def array2file(array, size, origin, spacing, imtype, fname):    
    image = itk.GetImageFromArray(array.reshape([size[2], size[1], size[0]]))
    image.SetSpacing((spacing[0], spacing[1], spacing[2]))
    image.SetOrigin((origin[0], origin[1], origin[2]))
    image2file(image, imtype=imtype, fname=fname)

# dataset of 3D image volume
# 3D volumes are resampled from and center-aligned with the original images
class Dataset(data.Dataset):
    def __init__(self, ids, rs_size, rs_spacing, rs_intensity, label_map, cls_num):
        self.ImageType = itk.Image[itk.SS, 3]
        self.LabelType = itk.Image[itk.UC, 3]
        self.ids = ids
        self.rs_size = rs_size
        self.rs_spacing = rs_spacing
        self.rs_intensity = rs_intensity
        self.label_map = label_map
        self.cls_num = cls_num
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        [d_name, casename, image_fn, image_ld_fn, label_fn] = self.ids[index]

        t = generate_transform(identity=True)

        src_image = read_image(fname=image_fn, imtype=self.ImageType)
        image = resample(
                    image=src_image, imtype=self.ImageType, 
                    size=self.rs_size, spacing=self.rs_spacing, origin=None, 
                    transform=t, linear=True, dtype=np.float32)
        image['array'] = normalize(image['array'], min=self.rs_intensity[0], max=self.rs_intensity[1])
        
        src_image_ld = read_image(fname=image_ld_fn, imtype=self.ImageType)
        image_ld = resample(
                      image=src_image_ld, imtype=self.ImageType, 
                      size=self.rs_size, spacing=self.rs_spacing, origin=None, 
                      transform=t, linear=True, dtype=np.float32)
        image_ld['array'] = normalize(image_ld['array'], min=self.rs_intensity[0], max=self.rs_intensity[1])
        
        src_label = read_image(fname=label_fn, imtype=self.LabelType)
        label = resample(
                    image=src_label, imtype=self.LabelType, 
                    size=self.rs_size, spacing=self.rs_spacing, origin=None, 
                    transform=t, linear=False, dtype=np.int64)

        tmp_array = np.zeros_like(label['array'])
        lmap = self.label_map[d_name]
        for key in lmap:
            tmp_array[label['array'] == key] = lmap[key]
        label['array'] = tmp_array
        label_bin = make_onehot(label['array'], cls=self.cls_num)
        label_exist = make_flag(cls=self.cls_num, labelmap=self.label_map[d_name])

        image_tensor = torch.from_numpy(image['array'])
        image_ld_tensor = torch.from_numpy(image_ld['array'])
        label_tensor = torch.from_numpy(label_bin)

        output = {}
        output['data'] = image_tensor
        output['data_ld'] = image_ld_tensor
        output['label'] = label_tensor
        output['label_exist'] = label_exist
        output['dataset'] = d_name
        output['case'] = casename
        output['size'] = image['size']
        output['spacing'] = image['spacing']
        output['origin'] = image['origin']
        output['org_size'] = image['org_size']
        output['org_spacing'] = image['org_spacing']
        output['org_origin'] = image['org_origin']
        output['eof'] = True

        return output

# dataset of image stacks (short-length 3D image volume)
# each image is resampled as a series adjacent image stacks
# the image stacks cover the whole range of image length
class DatasetStk(data.Dataset):
    def __init__(self, ids, rs_size, rs_spacing, rs_intensity, label_map, cls_num, perturb):
        self.ImageType = itk.Image[itk.SS, 3]
        self.LabelType = itk.Image[itk.UC, 3]
        self.ids = []
        self.rs_size = rs_size
        self.rs_spacing = rs_spacing
        self.rs_intensity = rs_intensity
        self.label_map = label_map
        self.cls_num = cls_num
        self.perturb = perturb

        for i, [d_name, casename, image_fn, image_ld_fn, label_fn] in enumerate(ids):

            print('Preparing image stacks ({}/{}) ...'.format(i, len(ids)))

            reader = sitk.ImageFileReader()
            reader.SetFileName(image_fn)
            reader.ReadImageInformation()

            size = reader.GetSize()
            spacing = reader.GetSpacing()
            origin = reader.GetOrigin()

            stack_len = rs_size[2] * rs_spacing[2]
            '''
            image_len = size[2] * spacing[2]
            stack_num = int(image_len / stack_len) + 1

            for stack_id in range(stack_num):
                stack_size = np.array(rs_size, dtype=int)
                stack_spacing = np.array(rs_spacing, dtype=float)
                stack_origin = np.zeros(3, dtype=float)
                stack_origin[0] = origin[0] + 0.5 * size[0] * spacing[0] - 0.5 * rs_size[0] * rs_spacing[0]
                stack_origin[1] = origin[1] + 0.5 * size[1] * spacing[1] - 0.5 * rs_size[1] * rs_spacing[1]
                #stack_origin[2] = origin[2] - 0.5 * (stack_num * stack_len - image_len) + stack_id * stack_len
                stack_perturb = np.zeros(2, dtype=float)
                if stack_num > 1:
                    stack_origin[2] = origin[2] + stack_id * (image_len - stack_len) / (stack_num - 1)
                    stack_perturb[0] = max(stack_origin[2] - 0.5 * stack_len, origin[2])
                    stack_perturb[1] = min(stack_origin[2] + 0.5 * stack_len, origin[2] + image_len - stack_len)
                else:
                    stack_origin[2] = origin[2] + 0.5 * (image_len - stack_len)
                    stack_perturb[0] = stack_origin[2]
                    stack_perturb[1] = stack_origin[2]
                self.ids.append([d_name, casename, image_fn, label_fn, stack_id, stack_size, stack_spacing, stack_origin, stack_perturb, stack_id == stack_num-1])
            '''

            lb_reader = sitk.ImageFileReader()
            lb_reader.SetFileName(label_fn)            
            lb_volume = lb_reader.Execute()
            lb_array = sitk.GetArrayFromImage(lb_volume)
            tmp_array = np.zeros_like(lb_array)
            lmap = self.label_map[d_name]
            for key in lmap:
                tmp_array[lb_array == key] = lmap[key]
            lb_array = tmp_array
            nz_ind = np.nonzero(lb_array > 0)
            lb_size = np.zeros(3, dtype=np.float)
            lb_origin = np.zeros(3, dtype=np.float)
            #for i in range(3):
            #    lb_size[i] = (np.max(nz_ind[2-i]) - np.min(nz_ind[2-i]) + 1) * spacing[i]
            #    lb_origin[i] = origin[i] + np.min(nz_ind[2-i]) * spacing[i]
            lb_size[2] = (np.max(nz_ind[0]) - np.min(nz_ind[0]) + 1) * spacing[2]
            lb_origin[2] = origin[2] + np.min(nz_ind[0]) * spacing[2]
            lb_len = lb_size[2]
            
            if lb_len > stack_len:
                stack_num = int((lb_len + stack_len) / stack_len) + 1
            else:
                stack_num = 1

            for stack_id in range(stack_num):
                stack_size = np.array(rs_size, dtype=int)
                stack_spacing = np.array(rs_spacing, dtype=float)
                stack_origin = np.zeros(3, dtype=float)
                stack_origin[0] = origin[0] + 0.5 * size[0] * spacing[0] - 0.5 * rs_size[0] * rs_spacing[0]
                stack_origin[1] = origin[1] + 0.5 * size[1] * spacing[1] - 0.5 * rs_size[1] * rs_spacing[1]
                #stack_origin[2] = origin[2] - 0.5 * (stack_num * stack_len - image_len) + stack_id * stack_len
                stack_perturb = np.zeros(2, dtype=float)
                if stack_num > 1:
                    stack_origin[2] = lb_origin[2] - 0.5 * stack_len + stack_id * lb_len / (stack_num - 1)
                    stack_perturb[0] = max(stack_origin[2] - 0.5 * stack_len, lb_origin[2] - 0.5 * stack_len)
                    stack_perturb[1] = min(stack_origin[2] + 0.5 * stack_len, lb_origin[2] + lb_len - 0.5 * stack_len)
                else:
                    stack_origin[2] = lb_origin[2] + 0.5 * (lb_len - stack_len)
                    stack_perturb[0] = lb_origin[2] - 0.5 * stack_len
                    stack_perturb[1] = lb_origin[2] + lb_len - 0.5 * stack_len
                self.ids.append([d_name, casename, image_fn, image_ld_fn, label_fn, stack_id, stack_size, stack_spacing, stack_origin, stack_perturb, stack_id == stack_num-1])
            
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        [d_name, casename, image_fn, image_ld_fn, label_fn, _, stack_size, stack_spacing, base_origin, stack_perturb, eof] = self.ids[index]

        stack_origin = base_origin.copy()
        if self.perturb:
            #stack_len = stack_size[2] * stack_spacing[2]
            #stack_origin[2] = stack_origin[2] + (random.random() - 0.5) * stack_len
            stack_origin[2] = stack_perturb[0] + random.random() * (stack_perturb[1] - stack_perturb[0])

        t = generate_transform(identity=True)

        src_image = read_image(fname=image_fn, imtype=self.ImageType)
        image = resample(
                    image=src_image, imtype=self.ImageType, 
                    size=stack_size, spacing=stack_spacing, origin=stack_origin, 
                    transform=t, linear=True, dtype=np.float32)        
        image['array'] = normalize(image['array'], min=self.rs_intensity[0], max=self.rs_intensity[1])
        
        src_image_ld = read_image(fname=image_ld_fn, imtype=self.ImageType)
        
        src_image_ld.SetOrigin(src_image.GetOrigin())
        src_image_ld.SetSpacing(src_image.GetSpacing())
        src_image_ld.SetDirection(src_image.GetDirection())
        image_ld = resample(
                    image=src_image_ld, imtype=self.ImageType, 
                    size=stack_size, spacing=stack_spacing, origin=stack_origin, 
                    transform=t, linear=True, dtype=np.float32)        
        image_ld['array'] = normalize(image_ld['array'], min=self.rs_intensity[0], max=self.rs_intensity[1])
        
        src_label = read_image(fname=label_fn, imtype=self.LabelType)
        
        src_label.SetOrigin(src_image.GetOrigin())
        src_label.SetSpacing(src_image.GetSpacing())
        src_label.SetDirection(src_image.GetDirection())
        label = resample(
                    image=src_label, imtype=self.LabelType, 
                    size=stack_size, spacing=stack_spacing, origin=stack_origin, 
                    transform=t, linear=False, dtype=np.int64)

        tmp_array = np.zeros_like(label['array'])
        lmap = self.label_map[d_name]
        for key in lmap:
            tmp_array[label['array'] == key] = lmap[key]
        label['array'] = tmp_array
#         if casename == 'case_00002':
#         print(casename,'\n')
#         print('label array\n')
#         print(np.max(label['array']), np.min(label['array']))
#         src_label_arr = itk.GetArrayFromImage(src_label)
#         print('src_label array\n')
#         print(np.max(src_label_arr), np.min(src_label_arr))
        
        label_bin = make_onehot(label['array'], cls=self.cls_num)
        label_exist = make_flag(cls=self.cls_num, labelmap=self.label_map[d_name])

        image_tensor = torch.from_numpy(image['array'])
        image_ld_tensor = torch.from_numpy(image_ld['array'])
        label_tensor = torch.from_numpy(label_bin)

        output = {}
        output['data'] = image_tensor
        output['data_ld'] = image_ld_tensor
        output['label'] = label_tensor
        output['label_exist'] = label_exist
        output['dataset'] = d_name
        output['case'] = casename
        output['size'] = image['size']
        output['spacing'] = image['spacing']
        output['origin'] = image['origin']
        output['org_size'] = image['org_size']
        output['org_spacing'] = image['org_spacing']
        output['org_origin'] = image['org_origin']
        output['eof'] = eof

        return output