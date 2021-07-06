import numpy as np
# Note:
# Use itk here will cause deadlock after the first training epoch 
# when using multithread (dataloader num_workers > 0) but reason unknown
import SimpleITK as sitk

def read_image(fname):
    reader = sitk.ImageFileReader()
    reader.SetFileName(fname)
    image = reader.Execute()
    return image

def resample_array(array, size, spacing, origin, size_rs, spacing_rs, origin_rs):
    array = np.reshape(array, [size[2], size[1], size[0]]).astype(dtype=np.uint8)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize((int(size_rs[0]), int(size_rs[1]), int(size_rs[2])))
    resampler.SetOutputSpacing((float(spacing_rs[0]), float(spacing_rs[1]), float(spacing_rs[2])))
    resampler.SetOutputOrigin((float(origin_rs[0]), float(origin_rs[1]), float(origin_rs[2])))
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    rs_image = resampler.Execute(image)
    rs_array = sitk.GetArrayFromImage(rs_image)

    return rs_array

def output2file(array, size, spacing, origin, fname):
    array = np.reshape(array, [size[2], size[1], size[0]]).astype(dtype=np.uint8)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(fname)
    writer.Execute(image)



################# init models ####################
def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
        layer.weight.data.normal_(0, math.sqrt(2. / n))
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()
    elif layer_name.find("Linear") != -1:
        layer.bias.data.zero_()

def init_model(net, restore=None, device=True):
    """Init models with cuda and weights."""
    if device==True:
        cudnn.benchmark = True
        net.cuda()
        net=nn.DataParallel(net) 
    ## init weights of model
#     net.apply(init_weights)
    
    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.module.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))
    return net


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def mkdir_if_missing(save_dir):
    if os.path.exists(save_dir):
        return 1
    else:
        os.makedirs(save_dir)
        return 0

############# save models ##################
def save_model(net, model_root, filename):
    """Save trained model."""
    flag = mkdir_if_missing(model_root)
    torch.save(net.module.state_dict(), os.path.join(model_root, filename))
