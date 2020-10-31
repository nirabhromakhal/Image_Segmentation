from keras.models import load_model, Model
from keras.layers import *
from keras_segmentation.models.unet import *
from keras_segmentation.predict import *
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf

# dataset details
NO_OF_SAMPLES = 1
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNELS = 3

# load weights
model = vgg_unet(n_classes=2, input_height=IMG_HEIGHT, input_width=IMG_WIDTH)

model.load_weights("tmp/vgg_unet_1.13")

# metrics
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return 1 - numerator / denominator


#model = load_model("tmp/vgg_unet_inria.h5", custom_objects={'dice_loss': dice_loss, 'mean_iou': mean_iou})

#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

# predict
test_dir = "Sample Dataset/test"
out_dir = "Sample Dataset/test_results"

#test_image = imread(test_dir + "/austin1_01_03.png")
#test_image = resize(test_image, output_shape=(1, 512, 512, 3))
#out = model.predict(test_image)

predict_multiple(model=model,
        inp_dir=test_dir,
        out_dir=out_dir,
        colors=[[0, 0, 0], [255, 255, 255]]
        )
'''
out = (out > 0.5).astype(np.uint8)
mask = out[0]
for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i][j] == 1:
                    mask[i][j] = 255
                else:
                    mask[i][j] = 0

np.set_printoptions(threshold=sys.maxsize)
print(mask)
imsave(fname=temp_dir + "/out.tif", arr=out[0])
plt.imshow(mask)
plt.show()
'''

