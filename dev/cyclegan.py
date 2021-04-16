
import os
import sys
import numpy as np
from glob import glob
from skimage.io import imread
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import *




#if not in_notebook():
#    import argparse
#    parser = argparse.ArgumentParser(description='MODEL ACTIVITY ANALYZER.')
#    parser.add_argument('--dataset', default='./dataset', type=str, help='path to dataset')
#    parser.add_argument('--model', default='model file name', type=str, help='model file name')
#    parser.add_argument('--lx', default=0, type=int, help='image length')
#    parser.add_argument('--ly', default=0, type=int, help='image width')
#    parser.add_argument('--n_sample', default=4, type=int, help='number of sample')
#    parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
#    parser.add_argument('--BS', default=32, type=int, help='number of epochs')
#    parser.add_argument('--prefix', default='', type=str, help='path to save the results')
##     parser.add_argument('--deep', default=0, type=int, help='Network depth!')
##     parser.add_argument('--dpi', default=200, type=int, help='image dpi')
#    parser.add_argument('--restart', action="store_true")

#    args = parser.parse_args()
#    data_path = args.dataset
#    lx,ly = args.lx,args.ly
#    n_sample = args.n_sample
#    restart = args.restart
#    EPOCHS = args.epochs
#    BS = args.BS
#    pp = args.pp
#    reg = args.reg
##     dpi = args.dpi
#    prefix = args.prefix+'/'
##     DEEP = args.deep
#else:
#    data_path = 'Alzheimer_dataset/train'
#    lx,ly = 128,128
#    n_sample = 4
#    restart = 0
#    EPOCHS = 1
#    BS = 32
##     dpi = args.dpi
#    prefix = 'alz1/'

data_path = '/home/vafaeisa/scratch/plank/'
lx,ly = 256,256
n_sample = 4
restart = 0
EPOCHS = 1
BS = 32
prefix = 'alz1/'


#paths = glob(data_path+'/*')
#fname = '{}-{}-{}'.format(data_path.split('/')[-1],lx,ly)
#if not os.path.isfile(fname+'.npz'):# or restart:
#    print("[INFO] reading data and preparation...")
#    x = []
#    labels = []
#    full_path = []
#    for path in paths:
#        files = glob(path+'/*')
#        for fil in files:
#            try:
#                img = imread(fil)
#                if lx*ly!=0:
#                    img = resize(img,output_shape=(lx,ly))
#                if img.ndim==3:
#                    img = np.mean(img,axis=-1)
#                x.append(img)
#                labels.append(fil.split('/')[-2])
#                full_path.append(fil)
#            except:
#                print('Something is wrong with',fil,', skipped.')
#    print("[INFO] prepared data is saved.")
#    np.savez(fname,x=x,labels=labels,full_path=full_path)
#    x = np.array(x)
#    labels = np.array(labels)
#else:
#    data = np.load(fname+'.npz'.format(lx,ly))
#    x = data['x']
#    labels = data['labels']
#    full_path = data['full_path']
#    print("[INFO] data is loaded...")


#int_map,lbl_map = int_label(labels)
#vec = [int_map[word] for word in labels]
#vec = np.array(vec)

#y = to_categorical(vec, num_classes=None, dtype='float32')
#x = x[:,:,:,None]/x.max()
#x = 2*x-1

## initialize the training data augmentation object
#trainAug = ImageDataGenerator(
#    rotation_range=5,
#    width_shift_range=0.03,
#    height_shift_range=0.03,
##   brightness_range=0.01,
##   shear_range=0.0,
#    zoom_range=0.03,
##   horizontal_flip=True,
##   vertical_flip=True,
#    fill_mode="nearest")
#describe_labels(y,verbose=1)
#x_us,y_us = balance_aug(x,y,trainAug)
## x_us,y_us = mixup(x,y,alpha=20,beta=1)
#describe_labels(y_us,verbose=1)
#x_us,y_us = shuffle_data(x_us,y_us)

#train_x0 = x_us[y_us[:,0].astype(bool)]
#train_x1 = x_us[y_us[:,1].astype(bool)]
#test_x0 = train_x0[:20]
#test_x1 = train_x1[:20]



#train_x0 = x_us[y_us[:,0].astype(bool)]
#train_x1 = x_us[y_us[:,1].astype(bool)]
#test_x0 = train_x0[:20]
#test_x1 = train_x1[:20]

def blocker(x,nside):
    xx = np.array_split(x, nside, axis=1)
    xx = np.concatenate(xx,axis=0)
    xx = np.array_split(xx, nside, axis=2)
    xx = np.concatenate(xx,axis=0)
    return xx


csep = 'healpix'
train_x0 = np.load(data_path+csep+'.npy')[:100]

csep = 'sevem'
train_x1 = np.load(data_path+csep+'.npy')[:100]

train_x0 = blocker(train_x0,8)
train_x1 = blocker(train_x1,8)


#fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,5))
#irr = np.random.randint(train_x0.shape[0])
#ax1.imshow(train_x0[irr],cmap='jet')
#ax2.imshow(train_x1[irr],cmap='jet')
#plt.tight_layout()
#plt.savefig('test.jpg')


train_x0 = train_x0/train_x0.max()
train_x0 = train_x0/train_x0.min()
train_x0 = 2*train_x0-1
train_x0 = train_x0[:,:,:,None]

train_x1 = train_x1/train_x1.max()
train_x1 = train_x1/train_x1.min()
train_x1 = 2*train_x1-1
train_x1 = train_x1[:,:,:,None]

test_x0 = train_x0[:20]
test_x1 = train_x1[:20]

print(train_x0.shape,train_x1.shape)

#input_img_size = (256, 256, 1)
input_img_size = train_x0.shape[1:]

buffer_size = 256
batch_size = 10

# Get the generators
gen_G = get_resnet_generator(input_img_size,name="generator_G")
gen_F = get_resnet_generator(input_img_size,name="generator_F")

# Get the discriminators
disc_X = get_discriminator(input_img_size,name="discriminator_X")
disc_Y = get_discriminator(input_img_size,name="discriminator_Y")

# Loss function for evaluating adversarial loss
adv_loss_fn = keras.losses.MeanSquaredError()

# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5

# Create cycle gan model
cycle_gan_model = CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
)

# Compile the model
cycle_gan_model.compile(
    gen_G_optimizer=keras.optimizers.Adam(learning_rate=5e-5, beta_1=0.5),
    gen_F_optimizer=keras.optimizers.Adam(learning_rate=5e-5, beta_1=0.5),
    disc_X_optimizer=keras.optimizers.Adam(learning_rate=5e-5, beta_1=0.5),
    disc_Y_optimizer=keras.optimizers.Adam(learning_rate=5e-5, beta_1=0.5),
    gen_loss_fn=generator_loss_fn,
    disc_loss_fn=discriminator_loss_fn,
)

# fake_train_x1 = self.gen_G(real_train_x0)
# fake_train_x0 = self.gen_F(real_train_x1)
cycle_gan_model.fit(train_x0, train_x1,
                    batch_size=10,
                    epochs=100,
#    callbacks=[plotter, model_checkpoint_callback],
                    )

cycle_gan_model.saveit('model1/')

cycle_gan_model.loadit('model1/')

_, ax = plt.subplots(4, 2, figsize=(10, 15))
#for i, img in enumerate(test_horses.take(4)):
for i in range(4):
    img = test_x0[i:i+1]
    prediction = np.array(cycle_gan_model.gen_G(img, training=False)[0])
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    img = (img[0] * 127.5 + 127.5).astype(np.uint8) #.numpy().astype(np.uint8)

    ax[i, 0].imshow(img)
    ax[i, 1].imshow(prediction)
    ax[i, 0].set_title("Input image")
    ax[i, 0].set_title("Input image")
    ax[i, 1].set_title("Translated image")
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")

    prediction = keras.preprocessing.image.array_to_img(prediction)
    prediction.save("predicted_img_{i}.png".format(i=i))
plt.tight_layout()
plt.show()

plt.savefig('fig.jpg',dpi=150)




