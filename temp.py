# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#C:\Users\psanthi\.cache\torch\hub\checkpoints
import cv2
import torch
import numpy as np
from building_footprint_segmentation.seg.binary.models import ReFineNet
from building_footprint_segmentation.helpers.normalizer import min_max_image_net
from building_footprint_segmentation.utils.py_network import (
    to_input_image_tensor,
    add_extra_dimension,
    convert_tensor_to_numpy,
    load_parallel_model
)
from building_footprint_segmentation.utils.operations import handle_image_size
from torch.utils import model_zoo

MAX_SIZE = 256
#MODEL_URL = "https://github.com/fuzailpalnak/building-footprint-segmentation/releases/download/alpha/refine.zip"
MODEL_URL = r"C:\Projects\building-footprint-segmentation-main\checkpoints\refine.zip"
blank_image = np.zeros((MAX_SIZE,MAX_SIZE,3), np.uint8)

def get_model():
    refine_net = ReFineNet()
    state_dict = model_zoo.load_url(MODEL_URL, progress=True, map_location="cpu")
    refine_net.load_state_dict(state_dict)
    return refine_net


def extract(original_image, model):

    original_height, original_width = original_image.shape[:2]

    if (original_height, original_width) != (MAX_SIZE, MAX_SIZE):
        original_image = handle_image_size(original_image, (MAX_SIZE, MAX_SIZE))

    # Apply Normalization
    normalized_image = min_max_image_net(img=original_image)

    tensor_image = add_extra_dimension(to_input_image_tensor(normalized_image))

    with torch.no_grad():
        # Perform prediction
        prediction = model(tensor_image)
        prediction = prediction.sigmoid()
    
    prediction_binary = convert_tensor_to_numpy(prediction[0]).reshape(
        (MAX_SIZE, MAX_SIZE)
    
    )


    prediction_3_channels = cv2.cvtColor(prediction_binary, cv2.COLOR_GRAY2RGB)
    
    print (prediction_3_channels[0])
        
    dst = cv2.addWeighted(
        blank_image,
        1,
        (prediction_3_channels * (255,69,0)).astype(np.uint8),
        0.4,
        0,
    )
   # cv2.imshow ("test", prediction_3_channels)
   # cv2.waitKey(0)
    
    cv2.imwrite (r"C:\Projects\Capture_pre.jpg", dst)

    return prediction_binary, prediction_3_channels, dst


def run(image_path):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    model = get_model()
    # PARALLELIZE the model if gpu available
    # model = load_parallel_model(model)
    
    prediction_binary, prediction_3_channels, dst = extract(original_image, model)
    return prediction_3_channels
    
   
dst = run (r"C:\Projects\Capture.jpg")
 
