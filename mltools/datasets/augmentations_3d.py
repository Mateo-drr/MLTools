import random

import numpy as np
import torch
import torchvision.transforms.functional as tf
import copy

from PIL import Image, ImageDraw


def random_local_rotation(
        in_tensor, tlbl, radius: int = 16, p: float | int = 0.5
):
    """https://www.mdpi.com/2313-433X/9/2/46"""
    if random.random() < p:
        diameter = radius * 2
        x, y = tlbl.shape
        # get a random point to rotate. this point is the top corner of the square area to be rotated
        xcenter = random.randint(
            0, (x - diameter)
        )  # random.choice(idk)#np.random.randint(0,32)#random.randint(0,x-diameter)
        ycenter = random.randint(
            0, (y - diameter)
        )  # random.choice(idk)#np.random.randint(0,32)#random.randint(0,y-diameter)
        ang = random.randint(20, 340)

        # Create the circle mask
        lum_img = Image.new(
            "L", [diameter, diameter], 0
        )  # create a square image size [diameter,diameter]
        draw = ImageDraw.Draw(lum_img)
        draw.pieslice(
            [(0, 0), (diameter - 1, diameter - 1)], 0, 360, fill=255
        )  # draw the circle in the image
        circmaks = torch.tensor(np.array(lum_img) / 255)
        # get surroundings of the circle crop but with the rotated mask
        invcircmask = (tf.rotate(circmaks.unsqueeze(0), ang) - 1) * -1

        # ROTATE LABEL
        # place the circle in the image
        circCrop = tlbl[xcenter : xcenter + diameter, ycenter : ycenter + diameter] * circmaks
        # rotate circular crop
        circCrop = tf.rotate(circCrop.unsqueeze(0), ang)
        # get surrounding crop
        invCircCrop = (
            tlbl[xcenter : xcenter + diameter, ycenter : ycenter + diameter] * invcircmask
        )
        # place the rotated circle back
        tlbl[xcenter : xcenter + diameter, ycenter : ycenter + diameter] = (
            circCrop + invCircCrop
        )

        # ROTATE CHUNK
        # place the circle in the image
        circCrop = (
                in_tensor[:, :, xcenter: xcenter + diameter, ycenter: ycenter + diameter] * circmaks
        )
        # rotate circular crop
        circCrop = tf.rotate(circCrop, ang)
        # get surrounding crop
        invCircCrop = (
                in_tensor[:, :, xcenter: xcenter + diameter, ycenter: ycenter + diameter]
                * invcircmask
        )
        # place the rotated circle back
        in_tensor[:, :, xcenter: xcenter + diameter, ycenter: ycenter + diameter] = (
            circCrop + invCircCrop
        )

    return in_tensor, tlbl


def cutmix(chunks, lbls, mask_size=32):
    """
    in_tensor shape [b,1,64,64,64]
    lbls shape [b,64,64]
    https://arxiv.org/pdf/1905.04899.pdf
    """
    idx1 = random.randint(0, chunks.shape[0] - 2)
    idx2 = idx1 + 1

    mask_size = (mask_size, mask_size, mask_size)
    # Randomly choose a position for the mask
    depth, height, width = chunks[idx1].shape[1:]
    d_start = random.randint(0, depth - mask_size[0])
    h_start = random.randint(0, height - mask_size[1])
    w_start = random.randint(0, width - mask_size[2])

    tchunk = copy.deepcopy(chunks[idx1])
    # put crop in chunk1
    chunks[idx1][
        :,
        d_start : d_start + mask_size[0],
        h_start : h_start + mask_size[1],
        w_start : w_start + mask_size[2],
    ] = chunks[idx2][
        :,
        d_start : d_start + mask_size[0],
        h_start : h_start + mask_size[1],
        w_start : w_start + mask_size[2],
    ]
    # put crop in chunk2
    chunks[idx2][
        :,
        d_start : d_start + mask_size[0],
        h_start : h_start + mask_size[1],
        w_start : w_start + mask_size[2],
    ] = tchunk[
        :,
        d_start : d_start + mask_size[0],
        h_start : h_start + mask_size[1],
        w_start : w_start + mask_size[2],
    ]

    tlbl = copy.deepcopy(lbls[idx1])
    # put crop in label1
    lbls[idx1][h_start : h_start + mask_size[1], w_start : w_start + mask_size[2]] = (
        lbls[idx2][h_start : h_start + mask_size[1], w_start : w_start + mask_size[2]]
    )
    # put crop in label2
    lbls[idx2][h_start : h_start + mask_size[1], w_start : w_start + mask_size[2]] = (
        tlbl[h_start : h_start + mask_size[1], w_start : w_start + mask_size[2]]
    )

    return chunks, lbls
