import random
import json
import pickle #for serialization
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import wordpunct_tokenize
from tensorflow import keras
from nltk.stem import WordNetLemmatizer #to identify similar type of word as one eg- work, working, worked, works etc as Work only

# Layers are the basic building blocks of neural networks in Keras,
# The core idea of Sequential API is simply arranging the Keras layers in a sequential order and so

from keras.models import Sequential

# The dense layer is a neural network layer that is connected deeply, which means each neuron in the dense layer receives input from all neurons of its previous layer
from keras.layers import Dense, Activation,Dropout
# optimization for gradient descent 
from keras.optimizers import SGD



lemmatizer = WordNetLemmatizer() #convert words into its base word of dictionary

intents = json.loads(open('D:\AI and ML\Projects\Chat-Bot\intent.json').read())

words=[]
classes=[]
documents=[]
ignore_letter=['?','!','.',',']

# Loading Data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_List = wordpunct_tokenize(pattern)
        words.extend(word_List)
        documents.append((word_List,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# print(documents)

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letter]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words,open('D:\AI and ML\Projects\Chat-Bot\words.pkl','wb'))
pickle.dump(classes,open('D:\AI and ML\Projects\Chat-Bot\classes.pkl','wb'))
# print(classes)
# print(words)

# Bag of word (converting word to number)
training =[]
output_empty = [0]*len(classes)

for document in documents:
    bag=[]
    word_patterns=document[0]
    word_patterns=[lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row=list(output_empty)
    output_row[classes.index(document[1])]=1
    training.append([bag,output_row])

# suffle data
random.shuffle(training)
training=np.array(training,dtype=object)

train_x=list(training[:,0])
train_y=list(training[:,1])

# building neural network
model=Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

sgd=SGD(learning_rate=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

hist=model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)
model.save('chatbotmodel.h5',hist)
print("Done")