from PIL import Image
import numpy as np
import os


class ImageManager:
    """ This class works with .fits files"""

    def __init__(self, image_folder_path, verbose=False):
        self.verbose = verbose
        self.files_path_arr = self._get_img_path_arr(image_folder_path)

    def _get_img_path_arr(self, fits_image_filepath):
        files_path_arr = []
        for path in os.listdir(fits_image_filepath):
            full_path = os.path.join(fits_image_filepath, path)
            if os.path.isfile(full_path):
                files_path_arr.append(full_path)
        return files_path_arr

    def _open_image(self, image_path, rgb_image):
        if rgb_image:
            image = Image.open(image_path)
        else:
            image = Image.open(image_path).convert('L')
        np_image = np.asarray(image)
        if self.verbose:
            print(image.format, image.size, image.mode)
        return np_image

    def get_images(self, rgb_image=True):
        arr_img = []
        for path in self.files_path_arr:
            arr_img.append(self._open_image(path, rgb_image))
        arr_img = np.array(arr_img)
        if self.verbose:
            print(f"The shape of the image out array is: {arr_img.shape}")

        return arr_img


def crop_image(in_image_path, out_image_path):
    manager = ImageManager(in_image_path)
    images = manager.get_images(rgb_image=True)
    cropped_images = images[:, 320:680, 50:950, :]
    iter = 0
    for im in cropped_images:
        new_image = Image.fromarray(im)
        new_image.save(out_image_path + f"/{iter}.png")
        iter += 1
    return False

