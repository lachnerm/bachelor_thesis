import functools

import numpy as np
import scipy.signal as convolve
from scipy.spatial.distance import hamming
from skimage.filters import gabor

from utils.gabor_kernel import gabor_kernel


def crop_center(img, crop_size, img_size):
    """
    Crops the center part of given size of an image. The image needs to be a 2d quadratic numpy array, which may also
    be flattened.

    :param img: image to be cropped
    :param crop_size: size of the center part that will be cropped
    :param img_size: size of the image
    :return: cropped image
    """
    img = img.reshape((img_size, img_size))
    start = img_size // 2 - (crop_size // 2)
    return img[start:start + crop_size, start:start + crop_size]


def calc_gabor_fhd(img1, img2, do_crop, img_size, crop_size, use_custom=False):
    """
    Computes the fractional hamming distance of two images after being transformed using a Gabor transformation and a
    binary transformation.

    :param img1: first image (2d numpy array)
    :param img2: second image (2d numpy array)
    :param do_crop: whether to crop the center part of the image
    :param img_size: size of the image
    :param crop_size: size of the center part that may be cropped
    :param use_custom: whether to use the second Gabor transformation
    :return: FHD of the two images
    """
    if use_custom:
        img1_bitstring = custom_gabor_transform(img1, do_crop, img_size, crop_size).flatten()
        img2_bitstring = custom_gabor_transform(img2, do_crop, img_size, crop_size).flatten()
    else:
        img1_bitstring = gabor_bit_transform(img1, do_crop, img_size, crop_size).flatten()
        img2_bitstring = gabor_bit_transform(img2, do_crop, img_size, crop_size).flatten()
    return hamming(img1_bitstring, img2_bitstring)


def gabor_bit_transform(img, do_crop, img_size, crop_size):
    """
    Applies the first Gabor transformation to an image and transforms all values to binary values using 0 as threshold.

    :param img: image to be transformed (2d numpy array)
    :param do_crop: whether to crop the center part of the image
    :param img_size: size of the image
    :param crop_size: size of the center part that may be cropped
    :return: gabor-transformed binary image
    """
    img = crop_center(img, crop_size, img_size) if do_crop else img.reshape((img_size, img_size))
    _, gabor_img = gabor(img, frequency=0.1)
    gabor_binary_img = (gabor_img >= 0).astype(np.uint8)
    return gabor_binary_img


def custom_gabor_transform(img, do_crop, img_size, crop_size):
    """
    Applies the second Gabor transformation to an image and transforms all values to binary values using 0 as threshold.

    :param img: image to be transformed (2d numpy array)
    :param do_crop: whether to crop the center part of the image
    :param img_size: size of the image
    :param crop_size: size of the center part that may be cropped
    :return: gabor-transformed binary image
    """
    img = crop_center(img, crop_size, img_size) if do_crop else img.reshape((img_size, img_size))
    gabor_img = convolve.convolve2d(img, gabor_kernel, mode='same', fillvalue=0, boundary='wrap')
    gabor_binary_img = (gabor_img >= 0).astype(np.uint8)
    return gabor_binary_img


def beautify_results_db_names(results):
    """
    Beautifies the dataset name of the results into a better representation for the visualization.

    :param results: results to be renamed
    :return: beautified results
    """
    plt_results = []
    for db_name, result in results:
        blocks = int(db_name.split("b")[0])
        mb = int(db_name.split("mb")[1].split("_")[0]) if len(db_name.split("mb")) > 1 else 0
        has_mb = mb > 0
        mb = "1/2" if blocks // 2 == mb else "2/3"
        every_2nd = "2nd" in db_name

        result_name = f"{blocks}{f'|m{mb}' if has_mb else ''}{'|2nd' if every_2nd else ''}"
        plt_results.append((result_name, result))

    return sorted(plt_results, key=functools.cmp_to_key(sort_db_names))


def sort_db_names(a, b):
    """
    Sorts the beautified database names by: number bits > whether type B > number of crps

    :param a: first database name
    :param b: second database name
    :return: a > b?
    """
    try:
        a = a[0]
        b = b[0]
        a_items = a.split("|")
        b_items = b.split("|")
        a_blocks = int(a_items[0])
        b_blocks = int(b_items[0])
        if a_blocks != b_blocks:
            return a_blocks - b_blocks

        if len(a_items) != len(b_items):
            return len(a_items) - len(b_items)

        if a_items[1] == "2nd":
            return -1
        elif b_items[1] == "2nd":
            return 1

        a_mb = a_items[1].split("m")[1]
        b_mb = b_items[1].split("m")[1]
        if a_mb != b_mb:
            return int(a_mb.split("/")[0]) - int(b_mb.split("/")[0])
        else:
            print("Cant compare ", a, " and ", b)
            exit()
    except Exception as e:
        print(e)
        print(a)
        print(b)
        exit()


def sort_raw_db_names(a, b):
    """
    Sorts the unbeautified database names by: number bits > whether type B > number of crps

    :param a: first database name
    :param b: second database name
    :return: a > b?
    """
    try:
        a_items = a.split("_")
        b_items = b.split("_")
        a_blocks = int(a_items[0].split("b")[0])
        b_blocks = int(b_items[0].split("b")[0])
        if a_blocks != b_blocks:
            return a_blocks - b_blocks

        if len(a_items) != len(b_items):
            return len(a_items) - len(b_items)

        if a_items[2] == "2nd":
            return -1
        elif b_items[2] == "2nd":
            return 1

        a_mb = a_items[2].split("mb")[1]
        b_mb = b_items[2].split("mb")[1]
        if a_mb != b_mb:
            return int(a_mb.split("/")[0]) - int(b_mb.split("/")[0])
        else:
            print("Cant compare ", a, " and ", b)
            exit()
    except Exception as e:
        print(e)
        print(a)
        print(b)
        exit()
