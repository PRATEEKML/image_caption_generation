from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image
import pickle
import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


# function to load object files
def load_files(path):
    ext=path.split('.')[-1]
    instance=None
    if ext=='pkl':
        with open(path,'rb') as f:
            instance=pickle.load(f)
    elif ext=='npy':
        instance= np.load(path,allow_pickle=True)
    return instance



# class for generating captions
class CaptionModel:
    def __init__(self):
        # mapping from word to index
        self.word_to_index = load_files('word_to_index.pkl')
        # mapping from index to word 
        self.index_to_word = load_files('index_to_word.pkl')
        # embedding layer weights
        self.embedding_idx = load_files('embedding_idx.npy')
        
        # total available words including special padding value
        self.vocab_size = len(self.word_to_index)+1
        # total number of lstm cells
        self.max_len = 33

        #Resnet convolutional base
        base_model=None
        if 'resnet50.h5' in os.listdir():
            base_model = ResNet50(weights='resnet50.h5')
        else:
            base_model= ResNet50(weights='imagenet')
        self.resnet_conv_base = Model(inputs = base_model.input, outputs = base_model.get_layer('avg_pool').output)

        #keras model for generating caption
        #image network
        img_input=Input((2048,))
        drop_img=Dropout(0.5)(img_input)
        img_act=Dense(256,activation='relu')(drop_img)
        #text network
        text_input=Input((self.max_len,))
        emb=Embedding(self.vocab_size, 50, mask_zero=True, weights=[self.embedding_idx], trainable=False)(text_input)
        drop_txt=Dropout(0.5)(emb)
        lstm=LSTM(256)(drop_txt)
        #final model after combing input from both layer
        combination=Add()([img_act,lstm])
        dense_1=Dense(256,activation='relu')(combination)
        dense_2=Dense(self.vocab_size, activation='softmax')(dense_1)
        self.word_predictor=Model(inputs=[img_input, text_input], outputs=dense_2)
        #initialising the model with trained weights
        self.word_predictor.load_weights('model_weight.h5')


    # method to make image dimensions as per resnet_50 standards
    def preprocess(self,img):
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)
        preds = self.resnet_conv_base.predict(x)
        return preds.flatten()


    # function for generating sentences from the preprocessed image vector
    def sentence_generator(self,photo):
        start = 'startseq'
        for i in range(self.max_len):
            sequence = [self.word_to_index[w] for w in start.split() if w in self.word_to_index]
            sequence = pad_sequences([sequence], maxlen=self.max_len)
            y_pred = self.word_predictor.predict([photo, sequence], verbose=0)
            y_pred = np.argmax(y_pred)
            word = self.index_to_word[y_pred]
            start += ' ' + word
            if word == 'endseq':
                break
        sents = start.split()
        sents = sents[1:-1]
        sents = ' '.join(sents)
        return sents


    #function to load image from the path and generate caption from it
    def predict_cation(self,path):
        img = image.load_img(path, target_size = (224,224))
        img= image.img_to_array(img,dtype='uint8')
        photo=self.preprocess(img)
        text=self.sentence_generator(photo.reshape((1,-1)))
        return text


#---------------------------------------------------------------------#