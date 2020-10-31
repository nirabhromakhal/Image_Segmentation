from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.train import train
from keras.models import load_model
import numpy as np
import os
from skimage.transform import *
from skimage.io import *

# dataset details
NO_OF_SAMPLES = 2000
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNELS = 3

model = vgg_unet(n_classes=2 ,  input_height=IMG_HEIGHT, input_width=IMG_WIDTH)
# model = load_model("tmp/vgg_unet_inria.h5")

train_images = "Sample Dataset/train"
train_annotations = "Sample Dataset/train_masks_01"

model.train(
    train_images=train_images,
    train_annotations=train_annotations,
    steps_per_epoch=200,
    auto_resume_checkpoint=True,
    checkpoints_path="tmp/vgg_unet_1",
    epochs=14
)

model.save(filepath="tmp/vgg_unet_inria.h5")