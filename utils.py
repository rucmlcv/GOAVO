# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
from six.moves import urllib

import numpy as np
import cv2
from PIL import Image

from inverse_warp import *


def compute_pairwise_loss(tgt_img,
                          ref_img,
                          tgt_depth,
                          ref_depth,
                          pose,
                          intrinsic,
                          with_mask=False):

    ref_img_warped, valid_mask, projected_depth, computed_depth = inverse_warp2(
        ref_img, tgt_depth, ref_depth, pose, intrinsic)

    diff_img = (tgt_img - ref_img_warped).abs()

    diff_depth = ((computed_depth - projected_depth).abs() /
                  (computed_depth + projected_depth).abs()).clamp(0, 1)

    # if args.with_ssim:
    #     ssim_map = (0.5*(1-ssim(tgt_img, ref_img_warped))).clamp(0, 1)
    #     diff_img = (0.15 * diff_img + 0.85 * ssim_map)

    if with_mask:
        weight_mask = (1 - diff_depth)
        diff_img = diff_img * weight_mask

    # compute loss
    # reconstruction_loss = mean_on_mask(diff_img, valid_mask)

    # geometry_consistency_loss = mean_on_mask(diff_depth, valid_mask)
    geometry_consistency_loss = torch.mean(diff_depth)

    # return reconstruction_loss, geometry_consistency_loss
    return geometry_consistency_loss


def cal_optical_flow(frame1, frame2):
    """Calculate the optical flow between two frames.
    """
    frame1 = np.array(frame1)
    frame2 = np.array(frame2)
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 5, 3, 5, 1.1,
                                        0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = Image.fromarray(rgb.astype('uint8')).convert('RGB')

    return rgb


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
        ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
         "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
        ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
         "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
        ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
         "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
        ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
         "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
        ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
         "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
        ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
         "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
        ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
         "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
        ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
         "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
        ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
         "cdc5fc9b23513c07d5b19235d9ef08f7"),
    }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum,
                                      model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path +
                                                                 ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum,
                                      model_path + ".zip"):
            print(
                "   Failed to download a file which matches the checksum - quitting"
            )
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))
