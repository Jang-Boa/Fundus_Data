import torch
import tqdm
from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from fundusutils.resnet import ResNet
from fundusutils.unetdecoder import UNet
from fundusutils.process import extract_mask, crop_with_mask


def get_preprocessing(resize):
    _transform = [
        A.Resize(resize, resize, p=1),
        A.Normalize(mean=0.0, std=1.0, max_pixel_value=255),
        ToTensorV2(),
    ]
    return A.Compose(_transform)

def get_vessel_mask(image_path, resize_scale=512):
    encoder_model = ResNet()
    model = UNet(encoder_model)
    model.eval()

    checkpoint = torch.load("./weights/ImFun_Vessel_Weight.pth", map_location='cpu')#['model_state']
    model.load_state_dict(checkpoint, strict=False)

    image, bg = extract_mask(image_path)
    image, _ = crop_with_mask(image, bg)
    transform = get_preprocessing(resize_scale)
    sample = transform(image=image)
    image_ = sample['image']
    image_ = image_.unsqueeze(dim=0)
    
    with torch.no_grad():
        predict = model(image_)
        predict = torch.where(predict>0.5, 1, 0)
        pred_mask = predict.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
        
    return image, pred_mask