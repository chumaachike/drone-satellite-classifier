from PIL import Image

def degrade_resolution(img, scale_factor):
    w, h = img.size

    new_w = max(1, w // scale_factor)
    new_h = max(1, h // scale_factor)

    small = img.resize(
        (new_w, new_h),
        resample=Image.BICUBIC
    )

    restored = small.resize(
        (w, h),
        resample=Image.BICUBIC
    )

    return restored