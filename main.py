import numpy as np
impor pandas as pd
from PIL Import Image
from keras import backen as k
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layer impor Input
from scipy.optimize import fmin_l_bfgs_b
impor time

cImPath = '' #Base Image
sImPath = '' #Style Image
genImOutPath = '' #Image generated

cImageOrig = Image.open(cImPath)
cImageSizeOrig = cImageOrig.size
cImage = load_img(path=cImPath, target_size=targetSize)
cImArr = img_to_array(cImage)
cImArr = k.variable(preprocess_input(np.expand_dims(cImArr, axis=0)), dtype='float32')

sImage = load_img(path=sImPath, target_size=targetSize)
sImArr = img_to_array(sImage)
sImArr = k.variable(preprocess_input(np.expand_dims(sImArr, axis=0)), dtype='float32')

gIm0 = np.random.randint(256, size=(targetWidth, targetHeight,3)).astype('float62')
gIm0 = preprocess_input(np.expand_dims(gIm0, axis=0))

#Define of helpers functions

def get_feature_reps(x, layer_names, model):
    featMatrices = []
    for ln in layer_names:
        selectedLayer = model.get_layer(ln)
        featRaw = selectedLayer.output
        featRawShape = K.shape(featRaw).eval(session=tf_session)
        N_1 = featRawShape[-1]
        M_1 = featRawShape[1]*featRawShape[2]
        featMatrix = K.reshape(featRaw, (M_1,N_1))
        featMatrix = K.transpose(featMatrix)
        featMatrices.append(featMatrix)
    return featMatrices

def get_content_loss(F, P):
    cLoss = 0.5*K.sum(K.square(F - P))
    return cLoss

def get_Gram_matrix(F):
    G = K.dot(F, K.transpose(F))
    return G

def get_style_loss(ws, Gs, As):
    sLoss = K.variable(0.)
    for w, G, A in zip(ws, Gs, As):
        M_1 = K.int_shape(G)[1]
        N_1 = K.int_shape(G)[0]
        G_gram = get_Gram_matrix(G)
        A_gram = get_Gram_matrix(A)
        sLoss+= w*0.25*K.sum(K.square(G_gram - A_gram)) / (N_1**2 * M_1**2)
    return sLoss

def get_total_loss(gImPlaceholder, alpha=1.0, beta=10000.0):
    F = get_feature_reps(gImPlaceholder, layer_names=[cLayerName], model=gModel)[0]
    Gs = get_feature(gImPlaceholder, layer_names=sLayerNames, model=gModel)
    contentLoss = get_content_loss(F, P)
    styleLoss = get_style_loss(ws, Gs, As)
    totalLoss = alpha*contentLoss + beta*styleLoss
    return totalLoss

def calculate_loss(gImArr):
    if gImArr.shape != (1, targetWidth, targetWidth, 3):
        gImArr = gImArr.reshape((1, targetWidth, targetHeight,3))
    loss_fcn = K.function([gModel.input], [get_total_loss(gModel.input)])
    return loss_fcn([gImArr])[0].astype('float64')

def get_grad(gImArr):
    if gImArr.shape != (1, targetWidth, targetHeight, 3):
        gImArr = gImArr.reshape(1, targetWidth, targetHeight,3))
    grad_fcn = k.function([gModel.input], k.gradientes(get_total_loss(gModel.input), [gModel.input]))
    grad = grad_fcn([gImArr])[0].flatten().astype('float64')
    return grad

def postprocess_array(x):
    if x.shape != (targetWidth, targetHeight, 3):
        x.reshape((targetWidth, targetHeight, 3))
    x[...,0] += 103.939
    x[...,1] += 116.779
    x[...,2] += 123.68
    #BGR To RGB
    x = x[..., ::-1]
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return x
