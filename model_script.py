from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image
import pickle
import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
#--------------------------------------------------------------------------------------------------------------


dir='./'

#mapping from word to index
with open(os.path.join(dir,'word_to_index.pkl'),'rb') as f:
	word_to_index=pickle.load(f)

#mapping from index to word 
with open(os.path.join(dir,'index_to_word.pkl'),'rb') as f:
    index_to_word=pickle.load(f)

#embedding layer weights
embedding_idx=np.load(os.path.join(dir,'embedding_idx.npy'),allow_pickle=True)
#total available words including special padding value
vocab_size=len(word_to_index)+1
max_len=33


#--------------------------------------------------------------------------------------------------------------


#Resnet convolutional base
base_model=None
if 'resnet50.h5' in os.listdir('./'):
    base_model = ResNet50(weights='resnet50.h5')
else:
    base_model= ResNet50(weights='imagenet')
rn = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

#keras model for generating caption

#image network
img_input=Input((2048,))
drop_img=Dropout(0.5)(img_input)
img_act=Dense(256,activation='relu')(drop_img)

#text network
text_input=Input((max_len,))
emb=Embedding(vocab_size,50,mask_zero=True,weights=[embedding_idx],trainable=False)(text_input)
drog_txt=Dropout(0.5)(emb)
lstm=LSTM(256)(drog_txt)

#final model after combing input from both layer
combination=Add()([img_act,lstm])
dense_1=Dense(256,activation='relu')(combination)
dense_2=Dense(vocab_size,activation='softmax')(dense_1)
model=Model(inputs=[img_input,text_input],outputs=dense_2)

#initialising the caption model with trained weights
model.load_weights('model_weight.h5')


#-------------------------------------------------------------------------------------------------------------


#making image fit for going in resnet_50 convolutional base
def preprocess(img):
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    preds = rn.predict(x)
    return preds.flatten()


#function for greedy searching of words and generating sentences
def caption(photo):
    start = 'startseq'
    for i in range(max_len):
        sequence = [word_to_index[w] for w in start.split() if w in word_to_index]
        sequence = pad_sequences([sequence], maxlen=max_len)
        y_pred = model.predict([photo,sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        word = index_to_word[y_pred]
        start += ' ' + word
        if word == 'endseq':
            break
    sents = start.split()
    sents = sents[1:-1]
    sents = ' '.join(sents)
    return sents


#function for generating caption
def gen_caption(path):
    img = image.load_img(path,target_size=(224,224))
    img= image.img_to_array(img,dtype='uint8')
    photo=preprocess(img)
    text=caption(photo.reshape((1,-1)))
    return text


#end-----------------------------------------------------------------------------------------------------------