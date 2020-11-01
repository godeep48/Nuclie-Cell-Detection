import numpy


def format_image(img):
    img = np.array(np.transpose(img, (1,2,0)))
    mean=np.array((0.485, 0.456, 0.406))
    std=np.array((0.229, 0.224, 0.225))
    img  = std * img + mean
    img = img*255
    img = img.astype(np.uint8)
    return img


def format_mask(mask):
    mask = np.squeeze(np.transpose(mask, (1,2,0)))
    return mask