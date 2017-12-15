# text_model
from keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense
from keras.models import Model


def fastText_model(l):
    input_layer = Input(shape=(l,), dtype='int32')
    embedding = Embedding(50001, 300, input_length=l)
    embed_seq = embedding(input_layer)
    fastText = GlobalAveragePooling1D()(embed_seq)
    dense_1 = Dense(64, activation='relu')(fastText)
    prediction = Dense(13, activation='sigmoid')(dense_1)
    model = Model(inputs=input_layer, outputs=prediction)
    return model
