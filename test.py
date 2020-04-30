#%%
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
import glob
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize

#%%
## point the varibles to Evaluation dataset
input_dir = './Datasets/LOLdataset/eval15/low/'
gt_dir = './Datasets/LOLdataset/eval15/high/'

#%%

im_height = 512
im_width = 512

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

model = load_model("orion_final.h5")

#%%

out = model.predict(input_images)

#%%
gt_images_tensor = tf.convert_to_tensor(gt_images)
out_tensor = tf.convert_to_tensor(out)
print("Evaluation SSIM:",np.mean(tf.image.ssim(gt_images_tensor,out_tensor,max_val=1.0)))
print("Evaluation PSNR:",np.mean(tf.image.psnr(gt_images_tensor,out_tensor,max_val=1.0)))

#%%

im1_in = resize(input_images[-1], (400, 600, 3), mode='constant', preserve_range=True)
im1 = resize(out[-1], (400, 600, 3), mode='constant', preserve_range=True)
im1_gt = resize(gt_images[-1], (400, 600, 3), mode='constant', preserve_range=True)
f,ax = plt.subplots(1,3,sharey=True, figsize=(24,18))
ax[0].imshow(im1_in)
ax[1].imshow(im1)
ax[2].imshow(im1_gt)
ax[0].set_title("Input Image")
ax[1].set_title('Orion Output')
ax[2].set_title("Ground Truth")
for a in ax:
  a.set_xticks([])
  a.set_yticks([])

plt.savefig("OUTPUT.png")
