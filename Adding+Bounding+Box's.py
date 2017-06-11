# coding: utf-8

# # First, lets explore the fisheries data

# In[1]:

import json
import os
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, InputLayer, Input, Flatten, Conv2D, MaxPooling2D, GlobalMaxPool2D, GlobalAveragePooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Activation

import PIL.Image

import os
from shutil import move , copy
from os.path import join
from os.path import split
path = join('data')
classes = os.listdir(join(path, 'train'))
classes


# In[2]:

boxes = os.listdir('./data/boxes/')
boxes


# In[3]:

classes = [c for c in classes if c[0] != '.']
classes


# In[4]:

import numpy as np
from numpy.random import choice

import glob

valid_ratio = 0.2
num_sample_images = 10
valid_path = join(path, 'valid')
if not os.path.exists(valid_path):
    os.makedirs(valid_path)

    for c in classes:
        cl_valid_path = join(valid_path, c)
        cl_train_path = join(path, 'train', c)
        cl_sample_path = join(path, 'sample', c)
        files = glob.glob(join(cl_train_path, '*.jpg'))
        os.makedirs(cl_valid_path)
        valid = choice(files, int(np.floor(0.2*len(files))), replace=False)
        [move(v, join(cl_valid_path, split(v)[1])) for v in valid]
        # We don't want the sample to contain the validation data, so redo the glob after moving.
        files = glob.glob(join(cl_train_path, '*.jpg'))
        os.makedirs(cl_sample_path)
        sample = choice(files, num_sample_images, replace=False)
        [copy(s, join(cl_sample_path, split(s)[1])) for s in sample]


# In[5]:

bs = 1
gen = ImageDataGenerator()
train_gen = gen.flow_from_directory(join(path,'train'), batch_size=bs, shuffle=False)
valid_gen = gen.flow_from_directory(join(path,'valid'), batch_size=bs, shuffle=False)
sample_gen = gen.flow_from_directory(join(path,'sample'), batch_size=bs, shuffle=False)


# In[6]:

import bcolz
from tqdm import tqdm
import os.path

def save_generator(gen, data_dir, labels_dir):
    """
    Save the output from a generator without loading all images into memory.

    Does not return anything, instead writes data to disk.

    :gen: A Keras ImageDataGenerator object
    :data_dir: The folder name to store the bcolz array representing the features in.
    :labels_dir: The folder name to store the bcolz array representing the labels in.
    :mode: the write mode. Set to 'a' for append, set to 'w' to overwrite existing data and 'r' to read only.

    """
    for directory in [data_dir, labels_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    num_samples = gen.samples

    d,l = gen.__next__()

    data = bcolz.carray(d, rootdir=data_dir, mode='w')
    labels = bcolz.carray(l, rootdir=labels_dir, mode='w')

    for i in tqdm(range(num_samples-1)):
        d, l = gen.__next__()
        data.append(d)
        labels.append(l)
    data.flush()
    labels.flush()


trn_data = join('bdat','train','data')
trn_label = join('bdat','train','label')
val_data = join('bdat','valid','data')
val_label = join('bdat','valid','label')
samp_data = join('bdat','sample','data')
samp_label = join('bdat','sample','label')

#save_generator(train_gen, trn_data, trn_label)
#save_generator(valid_gen, val_data, val_label)
#save_generator(sample_gen, samp_data, samp_label)

data = bcolz.open(trn_data)
labels = bcolz.open(trn_label)
val_data = bcolz.open(val_data)
val_labels = bcolz.open(val_label)


# In[7]:

import re
boxes


# In[8]:

re.findall('[a-zA-Z]*', boxes[0])


# To convert the coordinates to our resized size we will need to know the original image size.

# In[9]:

train_sizes = {f.split('\\')[-1] :PIL.Image.open(join(path,'train',f)).size for f in train_gen.filenames}
valid_sizes = {f.split('\\')[-1] :PIL.Image.open(join(path,'valid',f)).size for f in valid_gen.filenames}
sample_sizes = {f.split('\\')[-1] :PIL.Image.open(join(path,'sample',f)).size for f in sample_gen.filenames}


# In[10]:

all_sizes = {}
[all_sizes.update(d) for d in [train_sizes, valid_sizes, sample_sizes]]
all_sizes


# In[11]:

with open(join('./data/boxes/', boxes[0])) as fp:
    bxs = json.load(fp)
    print(bxs[0])
    print(bxs[0]['annotations'])


# So the annotations appear to be a list, the key parameter is the 'filename' parameter. We will also need to transform the coords as we have resized all of the images.

# A simple function to transform the coordinates of the bounding boxes.

# In[12]:

def convert_bb(item, size, new_size=(224,224)):
    factor_x = new_size[0]/size[0]
    factor_y = new_size[1]/size[1]
    item['height'] = item['height']*factor_y
    item['width'] = item['width']*factor_x
    item['x'] = item['x']*factor_x
    item['y'] = item['y']*factor_y
    return item



# Here we construct a dictionary whose keys are the filenames. The idea is we can use the filenames to find out the box meta-data.

# In[13]:

null_largest = {'width':0,
               'height': 0,
               'x': 224/2.,
               'y': 224/2.}

file2boxes = {}
for b in boxes:
    with open(join('./data/boxes/', b)) as fp:
        bxs = json.load(fp)
        blabel = re.findall('[a-zA-Z]*', b)[0]
        for item in bxs:
            it = item['annotations']

            for i in it:
                i['label'] = blabel

            fname = item['filename']

            if 'data' in fname:
                fname = os.path.split(fname)[1]
            existing = file2boxes.get(fname)

            item['annotations'] = [convert_bb(a, all_sizes[fname]) for a in item['annotations']]

            # Inplace sort so that we can find the largest box.
            item['annotations'].sort(key=lambda x: x['width']*x['height'])

            # Now lets just take the largest (last one)
            if len(item['annotations']) == 0:
                largest = null_largest
            else:
                largest = item['annotations'][-1]

            item['box_loc'] = [largest['width'], largest['height'], largest['x'], largest['y']]

            if existing is None:
                file2boxes[fname] = item
            else:
                file2boxes[fname].append(item['annotations'])



# Some of the files are unrepresented. Let's make sure they have a null largest.

# In[14]:

def get_boxes(gen, file2boxes=file2boxes, null_largest=null_largest):
    boxes = []
    nlarge =  [null_largest['width'], null_largest['height'], null_largest['x'], null_largest['y']]
    for f in gen.filenames:
        f = f.split('\\')[-1]
        meta = file2boxes.get(f)
        if meta == None:
            boxes.append(nlarge)
        else:
            largest = meta['box_loc']
            boxes.append(largest)
    return np.array(boxes)


# # Prepare the second output labels
#
# Now we have the labels, we need to line them up with the other data labels for train, valid and sample. The generators from each will help us with this.

# In[15]:

train_boxes = get_boxes(train_gen)
valid_boxes = get_boxes(valid_gen)
sample_boxes = get_boxes(sample_gen)


# # Let's use the previous simple model.

# In[16]:

simple = Sequential([
    InputLayer((256,256,3)),
    Conv2D(16,(3,3), activation='relu'),
    MaxPooling2D(4),
    Conv2D(16, (3,3), activation='relu'),
  ])
simple.summary()


# In[17]:

inp  = Input((256,256,3))
sim = simple(inp)
c1 = Conv2D(8,(3,3))(sim)
out1 = GlobalMaxPool2D()(c1)
c2 = Conv2D(4, (3,3))(sim)
out2 = GlobalMaxPool2D()(c2)
dualSimple = Model(input=inp, outputs=[out1, out2])


# In[18]:

dualSimple.compile(optimizer='adam',
              loss=['categorical_crossentropy', 'mse'],
              metrics=['accuracy'],
             loss_weights=[1,0.01])

dualSimple.summary()


# In[19]:

dualSimple.fit(data, [labels, train_boxes], epochs=3, batch_size=4, validation_data=(val_data, [val_labels, valid_boxes]))


# # Now Try with pretrained model + convolutions.

# In[20]:

conv_top = Sequential([InputLayer((8,8,2048)),
                       Conv2D(32, (3,3), padding='same', activation='relu'),
                       BatchNormalization(),
                      Conv2D(32, (3,3), padding='same', activation='relu'),
                      ])

inp = Input((8,8,2048))
ct = conv_top(inp)
x = Conv2D(8,(3,3), padding='same',activation='relu')(ct)
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
out1 = Activation('softmax')(x)

x = Conv2D(4,(3,3), padding='same',activation='relu')(ct)
x = BatchNormalization()(x)
out2 = GlobalAveragePooling2D()(x)

conv_model = Model(inputs=inp, outputs=[out1, out2])
conv_model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mse'],
                 metrics=['accuracy'], loss_weights=[1,0.001])

conv_model.summary()


# In[21]:

xCdata = bcolz.open(join('bdat','pretrained','data'))
xValdata = bcolz.open(join('bdat','pretrained','valdata'))


# In[22]:

conv_model.fit(xCdata, [labels, train_boxes], epochs=20, batch_size=8, validation_data=(xValdata, [val_labels, valid_boxes]))


# In[ ]:
