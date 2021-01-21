import numpy as np
from PIL import Image


''' 
Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array.
'''


def get_image_as_np_array(image_path, image_resize_size, center_crop_size, network_means, network_std_dev):
    im = Image.open(image_path)

    im = im.resize((image_resize_size, image_resize_size))

    im = im.crop(
        (
            (image_resize_size - center_crop_size) // 2,
            (image_resize_size - center_crop_size) // 2,
            (image_resize_size + center_crop_size) // 2,
            (image_resize_size + center_crop_size) // 2
        )
    )

    np_image = np.array(im) / 255

    np_image = (np_image - network_means) / network_std_dev

    np_image = np_image.transpose((2, 0, 1))

    return np_image
