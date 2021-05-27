
import numpy as np
import pandas as pd
import os
import tensorflow
import keras
import pickle
from keras.models import Model
from keras.layers import Input, LSTM, Dense,TimeDistributed,Embedding,Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from string import digits
import nltk
from nltk.stem import WordNetLemmatizer
import re
import string
import os
from .utils import Stemming

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', -1)

lines = pd.read_csv('bert_translate/BERTSQuADmaster/hindi_english_parallel.csv')
lines = lines[:10000]
lines.head()



# Lowercase all characters
lines['english']=lines['english'].apply(lambda x: str(x))
lines['hindi']=lines['hindi'].apply(lambda x: str(x))
lines['english']=lines['english'].apply(lambda x: x.lower())
lines['hindi']=lines['hindi'].apply(lambda x: x.lower())

lines['hindi'][0]

# Remove quotes
lines['english']=lines['english'].apply(lambda x: re.sub("'", '', x))
lines['hindi']=lines['hindi'].apply(lambda x: re.sub("'", '', x))

lines.head()

exclude = set(string.punctuation) # Set of all special characters
# Remove all the special characters
lines['english']=lines['english'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
lines['hindi']=lines['hindi'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

lines.head()

# Remove extra spaces
lines['english']=lines['english'].apply(lambda x: x.strip())
lines['hindi']=lines['hindi'].apply(lambda x: x.strip())
lines['english']=lines['english'].apply(lambda x: re.sub(" +", " ", x))
lines['hindi']=lines['hindi'].apply(lambda x: re.sub(" +", " ", x))

# Stemming and lemmatization
Stemmer = Stemming()
lines('english') = lines['english'].apply(lambda x: ' '.join(Stemmer.stem(Wd) for Wd in x.split()))
lines('english') = lines['english'].apply(lambda x: ' '.join(WordNetLemmatizer(Wd) for Wd in x.split()))

'hello! how are you buddy?'.strip()

lines['english'][0]

# Add start and end tokens to target sequences
lines['hindi'] = lines['hindi'].apply(lambda x : 'START_ '+ x + ' _END')

lines['hindi'][0]

### Get English and Hindi Vocabulary
all_eng_words=set()
for eng in lines['english']:
    for word in eng.split():
        if word not in all_eng_words:
            all_eng_words.add(word)

all_hindi_words=set()
for hin in lines['hindi']:
    for word in hin.split():
        if word not in all_hindi_words:
            all_hindi_words.add(word)

lines.head()

lines['length_eng']=lines['english'].apply(lambda x:len(x.split(" ")))
lines['length_hin']=lines['hindi'].apply(lambda x:len(x.split(" ")))

lines.head()
lines[lines['length_eng']>30].shape

lines=lines[lines['length_eng']<=20]
lines=lines[lines['length_hin']<=20]

print("maximum length of Hindi Sentence ",max(lines['length_hin']))
print("maximum length of English Sentence ",max(lines['length_eng']))

max_length_src=max(lines['length_hin'])
max_length_tar=max(lines['length_eng'])

input_words = sorted(list(all_eng_words))
target_words = sorted(list(all_hindi_words))
num_encoder_tokens = len(all_eng_words)
num_decoder_tokens = len(all_hindi_words)
num_encoder_tokens, num_decoder_tokens

num_decoder_tokens

num_decoder_tokens += 1

num_decoder_tokens

input_token_index = dict([(word, i+1) for i, word in enumerate(input_words)])
target_token_index = dict([(word, i+1) for i, word in enumerate(target_words)])

# input_token_index

reverse_input_char_index = dict((i, word) for word, i in input_token_index.items())
reverse_target_char_index = dict((i, word) for word, i in target_token_index.items())

# reverse_input_char_index

if not os.path.isfile("model_encoder.h5") or not os.path.isfile("model_decoder.h5"):

    from sklearn.model_selection import train_test_split
    X, y = lines['english'], lines['hindi']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=42)
    X_train.shape, X_test.shape

    X_train

    encoder_input_data = np.zeros((2, max_length_src),dtype='float32')
    decoder_input_data = np.zeros((2, max_length_tar),dtype='float32')
    decoder_target_data = np.zeros((2, max_length_tar, num_decoder_tokens),dtype='float32')

    def generate_batch(X = X_train, y = y_train, batch_size = 128):
        ''' Generate a batch of data '''
        while True:
            for j in range(0, len(X), batch_size):
                encoder_input_data = np.zeros((batch_size, max_length_src),dtype='float32')
                decoder_input_data = np.zeros((batch_size, max_length_tar),dtype='float32')
                decoder_target_data = np.zeros((batch_size, max_length_tar, num_decoder_tokens),dtype='float32')
                for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                    for t, word in enumerate(input_text.split()):
                        encoder_input_data[i, t] = input_token_index[word] # encoder input seq
                    for t, word in enumerate(target_text.split()):
                        if t<len(target_text.split())-1:
                            decoder_input_data[i, t] = target_token_index[word] # decoder input seq
                        if t>0:
                            decoder_target_data[i, t - 1, target_token_index[word]] = 1.
                yield([encoder_input_data, decoder_input_data], decoder_target_data)

    latent_dim = 300
    # Encoder
    encoder_inputs = Input(shape=(None,))
    enc_emb =  Embedding(num_encoder_tokens+1, latent_dim, mask_zero = True)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(num_decoder_tokens+1, latent_dim, mask_zero = True)
    dec_emb = dec_emb_layer(decoder_inputs)
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

    model.summary()
    train_samples = len(X_train)
    val_samples = len(X_test)
    batch_size = 128 
    epochs = 100

    model.fit_generator(generator = generate_batch(X_train, y_train, batch_size = batch_size),
                        steps_per_epoch = train_samples/batch_size,
                        epochs=epochs,
                        validation_data = generate_batch(X_test, y_test, batch_size = batch_size),
                        validation_steps = val_samples/batch_size)

    train_gen = generate_batch(X_train, y_train, batch_size = 1)


    # Encode the input sequence to get the "thought vectors"
    encoder_model = Model(encoder_inputs, encoder_states)

    # Decoder setup
    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    dec_emb2= dec_emb_layer(decoder_inputs) # Get the embeddings of the decoder sequence

    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2) # A dense softmax layer to generate prob dist. over the target vocabulary

    # Final decoder model
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs2] + decoder_states2)

    encoder_model.save('model_encoder.h5')
    decoder_model.save('model_decoder.h5')

def encode_sentence(sentence):
  encoder_input_data = np.zeros((1, max_length_src),dtype='float32')
  sentence = re.sub("[^a-zA-z\s]","", sentence)
  for t, word in enumerate(sentence.split()):
    encoder_input_data[0, t] = input_token_index[word]
  return encoder_input_data
                

def decode_sentence(sentence):
  decoder_input_data = np.zeros((1, max_length_tar),dtype='float32')
  for t, word in enumerate(sentence.split()):
    decoder_input_data[0, t] = target_token_index[word]
  return decoder_input_data

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or
           len(decoded_sentence) > 50):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

encoder_model = keras.models.load_model('model_encoder.h5', compile=False)
decoder_model = keras.models.load_model('model_decoder.h5', compile=False)


print(decode_sequence(encode_sentence("three ???....card deals"))[:-4])

def englishToHindiTranslate(sentence):
  return decode_sequence(encode_sentence(sentence))[:-4]





