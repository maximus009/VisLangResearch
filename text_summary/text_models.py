# text_model
from keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense
from keras.models import Model


def fastText_model(l=200, return_top=True):
    input_layer = Input(shape=(l,), dtype='int32')
    embedding = Embedding(50001, 300, input_length=l)
    embed_seq = embedding(input_layer)
    fastText = GlobalAveragePooling1D()(embed_seq)
    out_feature = Dense(64, activation='relu')(fastText)
    if return_top:
        out_feature = Dense(13, activation='sigmoid')(out_feature)

    model = Model(inputs=input_layer, outputs=out_feature)

    return model
