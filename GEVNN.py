from keras.models import Sequential
from keras.layers import Dense, Multiply, Lambda, Concatenate, Activation, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects
#from keras.optimizers import Adam as adam
from tensorflow.keras.optimizers import Adam as adam
from keras.losses import mse, binary_crossentropy
from keras import backend as K
from keras import regularizers
from sklearn.model_selection import train_test_split
import numpy as np

class Gev(Activation):

    def __init__(self, activation, **kwargs):
        super(Gev, self).__init__(activation, **kwargs)
        self.__name__ = 'gev'

def gev(x):
    return K.exp(-K.exp(-x))

get_custom_objects().update({'gev': Gev(gev)})


def distance_calc(tensors):
    euclid_dist = K.mean(K.square(tensors[0] - tensors[1]), axis=-1)
    euclid_dist = K.expand_dims(euclid_dist, 1)

    x1_val = K.sqrt(K.tf.reduce_sum(K.tf.matmul(tensors[0], K.transpose(tensors[0]))))
    x2_val = K.sqrt(K.tf.reduce_sum(K.tf.matmul(tensors[1], K.transpose(tensors[1]))))

    denom = K.tf.multiply(x1_val, x2_val)
    num = K.tf.reduce_sum(K.tf.multiply(tensors[0], tensors[1]), axis=1)

    cos_dist = K.tf.divide(num, denom) #.div
    cos_dist = K.expand_dims(cos_dist, 1)

    return [euclid_dist, cos_dist]

def distance_output_shape(input_shapes):
    shape1 = [input_shapes[0][0],1]
    shape2 = [input_shapes[1][0], 1]
    return [tuple(shape1), tuple(shape2)]

class AE:

    def __init__(self, trainX, valX, epoch_number, batch_size, learning_rate, encoder,
                 decoder, early_stopping, activation,
                 reg_lambda=0.0001, rand=0, verbose=0):
        if len(trainX.shape) < 2:
            trainX = np.expand_dims(trainX, axis=1)
        self.trainX = trainX
        self.valX = valX
        self.epoch = epoch_number
        self.batch = batch_size
        self.lr = learning_rate
        self.early_stopping = early_stopping
        self.encoder = encoder
        self.decoder = decoder
        self.array_size = trainX.shape[1]
        self.activation = activation
        self.reg_lambda = reg_lambda
        self.rand = rand
        self.verbose = verbose

    def AutoEncoder(self):

            trainX, valX = self.trainX, self.valX

            input_dim = self.array_size

            ### Auto-Encoder
            input_encoder = Input(shape=(input_dim,), name='input_encoder')

            encoded_output = Dense(self.encoder[0], activation='relu')(input_encoder)

            for layer in self.encoder[1:]:
                encoded_output = Dense(layer, activation='relu')(encoded_output)

            encoder = Model(input_encoder, encoded_output, name='encoder')

            input_dim_decoder = int(encoded_output.shape[1])
            input_decoder = Input(shape=(input_dim_decoder,), name='decoder_input')

            decoded_output = Dense(self.decoder[0], activation='relu')(input_decoder)

            for layer in self.decoder[1:]:
                decoded_output = Dense(layer, activation='relu')(decoded_output)

            decoded_output = Dense(self.array_size, activation='sigmoid')(decoded_output)
            decoder = Model(input_decoder, decoded_output, name='decoder_loss')

            outputs = decoder(encoder(input_encoder))

            ae = Model(input_encoder, outputs, name='AutoEncoder_loss')
            
            if self.verbose > 0:
                print(ae.summary())

            optimizer = adam(learning_rate=self.lr, epsilon=None, decay=0, amsgrad=False)

            ae.compile(loss='mean_squared_error', optimizer=optimizer)

            # fit network
            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=0, mode='auto')]

            ae.fit(trainX, trainX, epochs=self.epoch, batch_size=self.batch, verbose=self.verbose,
                            shuffle=False, validation_data=(valX, valX), callbacks=early_stop)

            decoder = ae
            encoder = Model(decoder.layers[0].input, decoder.layers[1].get_output_at(-1))

            return encoder, decoder
        
        
class MLP_AE:

    def __init__(self, trainX, trainY, epoch_number, batch_size, learning_rate, encoder,
                 decoder, sofnn, early_stopping, neurons, activation,
                 reg_lambda=0.0001, loss_weight=0.25, rand=0, verbose_ae=0, verbose_mlp=0):

        if len(trainX.shape)<2:
            trainX = np.expand_dims(trainX, axis=1)
        self.trainX = trainX
        self.trainY = trainY
        self.epoch = epoch_number
        self.batch = batch_size
        self.lr = learning_rate
        self.early_stopping = early_stopping
        self.neurons = neurons
        self.encoder = encoder
        self.decoder = decoder
        self.sofnn = sofnn
        self.array_size = trainX.shape[1]
        self.activation = activation
        self.reg_lambda = reg_lambda
        self.loss_weight = loss_weight
        self.rand=rand
        self.verbose_ae = verbose_ae
        self.verbose_mlp = verbose_mlp

    def data_preprocessing(self):

        trainX, valX, trainY, valY = train_test_split(self.trainX, self.trainY, test_size=0.20,
                                                        random_state=42)

        return trainX, valX, trainY, valY

    def MLP_AE(self):

            trainX, valX, trainY, valY = self.data_preprocessing()

            input_dim = self.array_size
            input_importance =Input(shape=(input_dim,), name='importance_input')
            input_encoder = Input(shape=(input_dim,), name='input_encoder')

            ### input selection neural network
            Input_selection = Sequential()
            Input_selection.add(Dense(self.sofnn[0], activation='tanh',
                                     input_dim=input_dim))

            if len(self.sofnn) > 1:
                for layer_size in self.sofnn[1:]:
                    Input_selection.add(Dense(layer_size, activation='tanh'))

            Input_selection.add((Dense(self.array_size, activation='softmax')))

            variable_importance = Input_selection(input_importance)
            importance = Model(input_importance, variable_importance, name='importance')

            selected_input = Multiply()([importance(input_importance), input_importance])

            ## auto_encoder

            ae = AE(trainX=self.trainX, valX= valX, epoch_number=self.epoch, batch_size=self.batch, learning_rate=self.lr,
                    encoder=self.encoder, decoder=self.decoder, early_stopping=self.early_stopping,
                    activation=self.activation, reg_lambda=self.reg_lambda, rand=self.rand, verbose=self.verbose_ae)

            encoder, decoder = ae.AutoEncoder()

            outputs = decoder(input_encoder)

            layer = Lambda(distance_calc, distance_output_shape)
            euclid_dist, cos_dist = layer([input_encoder, outputs])

            final_input = Concatenate()([euclid_dist, cos_dist, selected_input, encoder(input_encoder)])

            input_dim_mlp = int(final_input.shape[1])
            input_mlp = Input(shape=(input_dim_mlp,), name='input_mlp')

            Prediction = Dense(self.neurons[0], activation=self.activation,
                               kernel_regularizer=regularizers.l1(self.reg_lambda))(input_mlp)
            if len(self.neurons) > 1:
                for layer_size in self.neurons[1:]:
                    Prediction=Dense(layer_size, activation=self.activation,
                                     kernel_regularizer=regularizers.l1(self.reg_lambda))(Prediction)

            predicted_output = Dense(1, activation=self.activation)(Prediction)

            Prediction_MLP = Model(input_mlp, predicted_output, name='Prediction_loss')
            predicted_output = Prediction_MLP(final_input)

            final_model = Model(inputs=[input_importance, input_encoder], outputs=[outputs, predicted_output])
            
            if self.verbose_mlp > 0:
                print(final_model.summary())

            losses = {'AutoEncoder_loss': 'mean_squared_error',
                      'Prediction_loss': 'binary_crossentropy'}

            lossWeights = {"AutoEncoder_loss": self.loss_weight, "Prediction_loss": 1}


            optimizer = adam(learning_rate=self.lr, epsilon=None, decay=0, amsgrad=False)

            final_model.compile(loss=losses, loss_weights=lossWeights, optimizer=optimizer)

            # fit network
            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=0, mode='auto')
                          ]

            final_model.fit([trainX, trainX], [trainX, trainY], epochs=self.epoch, batch_size=self.batch, verbose=self.verbose_mlp,
                           shuffle=False, validation_data=([valX, valX], [valX, valY]), callbacks=early_stop)

        

            return final_model
        
    def predict(self, testX, testY, final_model):

        _, pred_Y = final_model.predict([testX, testX])

        return pred_Y, testY