import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import glob
from skimage.io import imread, imshow
from skimage.transform import resize
from model import *

#%%
## Change the input directory to point to the Dataset root
input_dir = './Datasets/LOLdataset/our485/low/'
gt_dir = './Datasets/LOLdataset/our485/high/'
checkpoint_dir = './result_orion/'
result_dir = './result_orion/'

im_height = 512
im_width = 512

#%%
def ssim_loss(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred,1.0))

#%%
gt_images = []
input_images = []
input_files = sorted(glob.glob(input_dir+'*.png'))
gt_files = sorted(glob.glob(gt_dir+'*.png'))
for in_path,gt_path in zip(input_files, gt_files):
    in_im = imread(in_path)
    gt_im = imread(gt_path)
    in_img = resize(in_im, (im_height, im_width, 3,1), mode='constant', preserve_range=True)
    gt_img = resize(gt_im, (im_height, im_width, 3,1), mode='constant', preserve_range=True)
    err = tf.losses.MAE(in_img, gt_img)#np.mean(np.abs(in_img-gt_img))
    in_img = in_img[:,:,:,0] * 4 + 25
    in_img = resize(in_img, (im_height, im_width, 3), mode='constant', preserve_range=True)
    gt_img = resize(gt_im, (im_height, im_width, 3), mode='constant', preserve_range=True)
    input_images.append(np.array(in_img/255.0,dtype='float32'))
    gt_images.append(np.array(gt_img/255.0,dtype='float32'))

input_images = np.array(input_images,dtype='float32')
gt_images = np.array(gt_images,dtype='float32')


#%%
input_img = Input((im_height, im_width,3),name='img')
model = orion_network(input_img)
model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_absolute_error", metrics=[ssim_loss])
#model.summary()

callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(monitor='loss', factor=0.01, patience=10, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-orion.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)
    ]

#%%
results = model.fit(input_images, gt_images, batch_size=1, epochs=4000, callbacks=callbacks)

#%%
model.save("orion_final.h5")
