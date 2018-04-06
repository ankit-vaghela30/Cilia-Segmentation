from keras.engine.topology import Input
from keras import initializers
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
import keras

'''
We used tiramisu model created by Jeremy Howard in his Fast.ai deep learning course. The model was extracted from
jupyter notebook which is located here: https://github.com/fastai/courses/blob/master/deeplearning2/tiramisu-keras.ipynb
This model is based on paper titled 'The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic
Segmentation' located here: https://arxiv.org/abs/1611.09326

'''

def relu(x): return Activation('relu')(x)

def dropout(x, p): return Dropout(p)(x) if p else x

def bn(x): return BatchNormalization()(x)

def relu_bn(x): return relu(bn(x))

def concat(xs): return merge(xs, mode='concat', concat_axis=-1)

def conv(x, nf, sz, wd, p, stride=1):
    x = Convolution2D(nf, sz, sz, init='he_uniform', border_mode='same',
                      subsample=(stride,stride), W_regularizer=l2(wd))(x)
    return dropout(x, p)

def conv_relu_bn(x, nf, sz=3, wd=0, p=0, stride=1):
    return conv(relu_bn(x), nf, sz, wd=wd, p=p, stride=stride)
def dense_block(n,x,growth_rate,p,wd):
    added = []
    for i in range(n):
        b = conv_relu_bn(x, growth_rate, p=p, wd=wd)
        x = concat([x, b])
        added.append(b)
    return x,added
def transition_dn(x, p, wd):
#     x = conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd)
#     return MaxPooling2D(strides=(2, 2))(x)
    return conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd, stride=2)
def down_path(x, nb_layers, growth_rate, p, wd):
    skips = []
    for i,n in enumerate(nb_layers):
        x,added = dense_block(n,x,growth_rate,p,wd)
        skips.append(x)
        x = transition_dn(x, p=p, wd=wd)
    return skips, added
def transition_up(added, wd=0):
    x = concat(added)
    _,r,c,ch = x.get_shape().as_list()
    return Deconvolution2D(ch, 3, 3, (None,r*2,c*2,ch), init='he_uniform',
               border_mode='same', subsample=(2,2), W_regularizer=l2(wd))(x)
#     x = UpSampling2D()(x)
#     return conv(x, ch, 2, wd, 0)
def up_path(added, skips, nb_layers, growth_rate, p, wd):
    for i,n in enumerate(nb_layers):
        x = transition_up(added, wd)
        x = concat([x,skips[i]])
        x,added = dense_block(n,x,growth_rate,p,wd)
    return x

def reverse(a): return list(reversed(a))

def create_tiramisu(nb_classes, nb_dense_block=6,
    growth_rate=16, nb_filter=48, nb_layers_per_block=5, p=None, wd=0):

    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else: nb_layers = [nb_layers_per_block] * nb_dense_block

    x = conv(Input((256,256,3)), nb_filter, 3, wd, 0)
    skips,added = down_path(x, nb_layers, growth_rate, p, wd)
    x = up_path(added, reverse(skips[:-1]), reverse(nb_layers[:-1]), growth_rate, p, wd)

    x = conv(x, nb_classes, 1, wd, 0)
    _,r,c,f = x.get_shape().as_list()
    y= Activation('softmax')(x)
    model= Model(Input((256,256,3)), y)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model
