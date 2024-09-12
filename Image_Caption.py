#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import re
import nltk
from nltk.corpus import stopwords
import string
import json
from time import time
import pickle
import cv2
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add


# In[2]:


# reading captions
def readTxtFile(path):
    with open(path) as f:
        captions = f.read()
    return captions


# In[3]:


captions = readTxtFile('../input/flickr8k-sau/Flickr_Data/Flickr_TextData/Flickr8k.token.txt')
captions = captions.split('\n')
captions = captions[:-1] # slicing as last line as it is empty
print(captions[0])
print(len(captions))


# In[4]:


# The format in which the captions are stored is img_id #caption_num caption. 
# Therefore we will create a mapping between captions in the list to its corresponding image.
descriptions = {}
for caption in captions:
    first,second = caption.split('\t') # Splitting the caption wrt '\t'
    image = first.split('.')[0]
    
    if descriptions.get(image) is None:
        descriptions[image] = []
    descriptions[image].append(second)


# In[5]:


descriptions["1000268201_693b08cb0e"]


# In[6]:


# Image to the corresponding description.
IMG_PATH = '../input/flickr8k-sau/Flickr_Data/Images/'
img = cv2.imread(IMG_PATH + "1000268201_693b08cb0e.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8,8))
plt.imshow(img)
plt.show()


# ## Cleaning captions
# 1. Converting to lower case.
# 2. Removing numbers.
# 3. Removing puntuations.
# 
# We will not remove stopwords and will not lemmatize text as we want proper sentences as our output

# In[7]:


# A function which will recieve sentence as an input and will return a clean_text.
def cleanText(sentence):
    sentence = sentence.lower()
    sentence = re.sub("[^a-z]+"," ",sentence) # Replacing which which are not formed by alphabets with a space
    sentence = sentence.split()
    sentence = [s for s in sentence if len(s) > 1] # removing word which has length 1
    sentence = " ".join(sentence) 
    return sentence


# In[8]:


# Cleaning all the captions
for key,caption_list in descriptions.items():
    for i in range(len(caption_list)):
        caption_list[i] = cleanText(caption_list[i])


# In[9]:


descriptions["1000268201_693b08cb0e"]


# In[10]:


# # As we may have to use the same data. We are going to store the above cleaned data to a text file
# with open("./description.txt",'w') as f:
#     f.write(str(descriptions))


# ## Creating vocab
# 1. Store all the unique word in form of dict.
# 2. We will discard words which will have a freq less than a threshold frequency.
# 
# 
#  This will reduce our size of vocab. 

# In[11]:


# descriptions = None
# with open("./description.txt",'r') as f:
#     descriptions = f.read()
    
#Converting descriptions frrom string to dictionary
# json_accepted = descriptions.replace("'","\"")
# descriptions = json.loads(json_accepted)
print(type(descriptions))


# In[12]:


# Vocab 
vocab = set()

for key in descriptions.keys():
    [vocab.update(sentence.split()) for sentence in descriptions[key]]

total_words = []

print("Vocab Size is: %d"%len(vocab)) 

for key in descriptions.keys():
    for sentence in descriptions[key]:
        for i in sentence.split():
            total_words.append(i)
            
print("Total Words: %d"%len(total_words))

# Datastructure to count the frequency of the words in all words
import collections
counter = collections.Counter(total_words)
freq_count = dict(counter)
sorted_freq_count = sorted(freq_count.items(),reverse=True,key = lambda x: x[1])

#filtering the vocab
threshold = 10

sorted_freq_count = [x for x in sorted_freq_count if x[1] > threshold]
total_words = [x[0] for x in sorted_freq_count]
print("Total Words after filtering: %d"%len(total_words))


# ## Preparing training and testing data
# 
# Format of data is as follows
# 1. All the images are in single folder "Images"
# 2. Captions are stored in the text file with corresponding image_name
# 3. Data is divided into train and test.
# 4. train.txt and text.txt contain image ids .
# 

# In[13]:


# reading the training and testing data
training_file_data = readTxtFile("../input/flickr8k-sau/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt")
test_file_data = readTxtFile("../input/flickr8k-sau/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt")

train = [row.split(".")[0] for row in training_file_data.split("\n")[:-1]]
test = [row.split(".")[0] for row in test_file_data.split("\n")[:-1]]


# In[14]:


# creating the dictionary for training data
train_description = {}

# attaching startseq and endseq with each caption to prepare training data
for image_id in train:
    train_description[image_id] = []
    for caption in descriptions[image_id]:
        caption = "startseq " + caption + " endseq"  
        train_description[image_id].append(caption)
        
print(train_description["1000268201_693b08cb0e"])


# ## Image preprocessing
# 
# 1. For freature extraction of image we have use ResNet50 trained on imagenet dataset with skip connections

# In[15]:


# Model creation
model = ResNet50(weights="imagenet",input_shape=(224,224,3))
model.summary()


# In[16]:


# we will use the output from the Global Average Pooling layer of the ResNet50
model_new = Model(model.input,model.layers[-2].output)


# In[17]:


#Image pre-processing
from keras.preprocessing import image
def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    # As the input to a ResNet is a 4D tensor.Therefore expanding the size (1,224,224,3)
    img = np.expand_dims(img,axis=0) 
     #Normalisation
    img = preprocess_input(img)    
    return img

img = preprocess_img(IMG_PATH+"1000268201_693b08cb0e.jpg")
plt.imshow(img[0])
plt.axis("off")
plt.show()


# In[18]:


def encode_image(img):
    img = preprocess_img(img)
    feature_vector = model_new.predict(img)
    feature_vector = feature_vector.reshape((-1,))
    return feature_vector


# In[19]:


encode_image(IMG_PATH+"1000268201_693b08cb0e.jpg")


# In[20]:


#Encoding every image in the training dataset ie. extracting context vector from the ResNet for the training data.
encoding_train = {}
start = time()
for ix,img_id in enumerate(train):
    img_path = IMG_PATH + img_id + ".jpg"
    encoding_train[img_id] = encode_image(img_path)
    
    if ix%100 == 0:
        print("Progress of encoding timestep %d"%ix)
        
end = time()
print("Total time taken %f" %(end-start))


# In[21]:


# Encoding every test image. ie extracting context vector from the ResNet50 for testing purpose
encoding_test = {}
start = time()
for ix,img_id in enumerate(test):
    img_path = IMG_PATH + img_id + ".jpg"
    encoding_test[img_id] = encode_image(img_path)
    
    if ix%100 == 0:
        print("Progress of encoding timestep %d"%ix)
        
end = time()
print("Total time taken %f" %(end-start))


# In[22]:


get_ipython().system('mkdir saved')


# In[23]:


with open("saved/encoding_train_features.pkl","wb") as f:
    pickle.dump(encoding_train,f)


# In[24]:


with open("saved/encoding_test_features.pkl","wb") as f:
    pickle.dump(encoding_test,f)


# ## Data preprocessing for captions
# 1. Making dictionaries for word to index and for index to words

# In[26]:


word_to_index = {}
index_to_word = {}

for i,word in enumerate(total_words):
    # 0th index is reserved for the words which are not present in out vocab.
    word_to_index[word] = i+1
    index_to_word[i+1] = word

# Adding two additional words 
index_to_word[1846] = "startseq"
word_to_index["startseq"] = 1846
index_to_word[1847] = "endseq"
word_to_index["endseq"] = 1847

vocab_size = len(word_to_index) + 1


# In[27]:


# Now we will find the maximum length of the caption. As this will be one of the dimension of the input.

max_len = 0
for key in train_description.keys():
    for caption in train_description[key]:
        max_len = max(max_len,len(caption.split()))

print(max_len)


# In[28]:


with open("saved/word_to_index.pkl","wb") as f:
    pickle.dump(word_to_index,f)

with open("saved/index_to_word.pkl","wb") as f:
    pickle.dump(index_to_word,f)


# ## Data Generator
# 1. We will train our model where data is feeded in batch.

# In[29]:


def data_generator(train_descriptions,encoding_train,word_to_idx,max_len,batch_size):
    X1,X2,y = [],[],[]
    n = 0
    while True:
        for key,desc_list in train_descriptions.items():
            n += 1
            
            photo = encoding_train[key]
            for desc in desc_list:
                
                seq = [word_to_idx[word] for word in desc.split() if word in word_to_idx]
                for i in range(1,len(seq)):
                    xi = seq[0:i]
                    yi = seq[i]
                    
                    #0 denote padding word
                    xi = pad_sequences([xi],maxlen=max_len,value=0,padding='post')[0]
                    yi = to_categorical([yi],num_classes=vocab_size)[0]
                    
                    X1.append(photo)
                    X2.append(xi)
                    y.append(yi)
                    
                if n == batch_size:
                    yield ([np.array(X1),np.array(X2)],np.array(y))
                    X1,X2,y = [],[],[]
                    n = 0


# ## Using glove embedding to embedd captions

# In[30]:


f = open("../input/glove6b/glove.6B.50d.txt",encoding='utf8')
embedding_index = {}

for line in f:
    values = line.split()
    word = values[0]
    word_embedding = np.array(values[1:],dtype='float')
    embedding_index[word] = word_embedding

f.close()

print(embedding_index["abs"])


# In[31]:


#we will create our embedding matrix with words which are present in out vocab

def get_embedding_matrix():
    emb_dim = 50
    matrix = np.zeros((vocab_size,emb_dim))
    for word,idx in word_to_index.items():
        embedding_vec = embedding_index.get(word)
        if embedding_vec is not None:
            matrix[idx] = embedding_vec
    return matrix


# In[32]:


embedding_matrix = get_embedding_matrix()
print(embedding_matrix.shape)


# ## Model Architecture

# In[44]:


# Taking image inputs
input_img_feature = Input(shape = (2048,))
inp_img1 = Dropout(0.2)(input_img_feature)
inp_img2 = Dense(256,activation = 'relu')(inp_img1)

#Taking Caption input

input_captions = Input(shape = (max_len,))
inp_cap1 = Embedding(input_dim = vocab_size,output_dim = 50, mask_zero = True, weights=[embedding_matrix], trainable=False)(input_captions)
inp_cap2 = Dropout(0.2)(inp_cap1)
inp_cap3 = LSTM(256, dropout=0.2)(inp_cap2)

decoder1 = add([inp_img2,inp_cap3])
decoder2 = Dense(256,activation = 'relu')(decoder1)
outputs = Dense(vocab_size,activation = 'softmax')(decoder2)

# Combining models

model = Model(inputs = [input_img_feature,input_captions],outputs = outputs)
model.summary()


# ## Model Training

# In[45]:


epochs = 20
batch_size = 64
steps = len(train_description)//batch_size

model.compile(loss='categorical_crossentropy',optimizer= 'adam')

def train():
    for i in range(epochs):
        generator = data_generator(train_description,encoding_train,word_to_index,max_len,batch_size)
        model.fit_generator(generator,epochs=1, steps_per_epoch=steps, verbose=1)
        model.save('./model_weights/model_'+str(i)+'.h5')
        
train()


# In[46]:


# Predicting output
def predict_caption(photo):
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_to_index[w] for w in in_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence],maxlen = max_len,padding='post')
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax() #greedy sampling of the words.
        word = index_to_word[ypred]
        in_text += ' ' + word
        
        if word == "endseq":
            break
    final_caption = in_text.split()[1:-1] # Removing startseq and endseq from the captions
    final_caption = ' '.join(final_caption)
    return final_caption


# In[47]:


# Picking random images and seeing result

for i in range(15):
    idx = np.random.randint(0,1000)
    all_img_names = list(encoding_test.keys())
    img_name = all_img_names[idx]
    photo_2048 = encoding_test[img_name].reshape((1,2048))
    i = plt.imread(IMG_PATH+img_name+".jpg")
    caption = predict_caption(photo_2048)
    plt.title(caption)
    plt.imshow(i)
    plt.axis("off")
    plt.show()


# ## Evaluation of Results

# ### BLEU Metric

# In[48]:


def BLEU_Score(y_test, y_pred):
    references = [[ele.split() for ele in y_test]]
    candidates = [y_pred.split()]
    return corpus_bleu(references, candidates)


# In[49]:


from nltk.translate.bleu_score import corpus_bleu
scores = []

for idx in range(len(list(encoding_test.keys()))):
    all_img_names = list(encoding_test.keys())
    img_name = all_img_names[idx]
    photo = encoding_test[img_name].reshape((1,2048))
    i = plt.imread(IMG_PATH+img_name+".jpg")
    caption = predict_caption(photo)
    scores.append(BLEU_Score(descriptions[img_name], caption))


# In[50]:


np.mean(scores)

