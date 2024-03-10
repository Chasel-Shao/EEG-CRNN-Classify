from keras.layers import Dense, Activation, Dropout, Conv1D, LSTM, MaxPooling1D
from keras.layers import Flatten, Conv2D, MaxPooling2D, GRU, BatchNormalization
from keras.layers import Conv3D, MaxPool3D, Reshape, Input, AveragePooling2D
from keras.models import Sequential, load_model, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers.legacy import Adam
from keras.regularizers import l2
from keras.losses import CategoricalCrossentropy
from utils import *
import tensorflow as tf
from tensorflow.keras.regularizers import L1L2


class SequentialModel:
    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, config):
        """ Virtual Function """
        return

    def train(self, x, y, x_val, y_val, config, save_dir):
        """ Virtual Function """
        return

    def evaluate(self, x_test, y_test, verbose=1):
        return self.model.evaluate(x_test,  y_test, verbose=verbose)

    def predict(self, x_test, verbose=1):
        return self.model.predict(x_test, verbose=verbose)

    
class VanillaRNN(SequentialModel):
    def __init__(self):
        super(VanillaRNN, self).__init__()

    def build_model(self, config):
        # replace hardcoded dimensions with config dictionary
        model = self.model
        if config['LSTM']:
            model.add(LSTM(22, input_shape=(config['input_shape'][0], config['input_shape'][1]), return_sequences=True))
        else:
            model.add(GRU(22, input_shape=(
            config['input_shape'][0], config['input_shape'][1]),
                           return_sequences=True))
        model.add(Flatten())
        model.add(Dropout(config['dropout']))
        model.add(Dense(4, activation='softmax'))
        optimizer = Adam(learning_rate=config['lr'], beta_1=0.9, beta_2=0.999,
                         amsgrad=False)
        model.compile(optimizer=optimizer,
                      loss=CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.summary()
        print("Model compiled.")

    def train(self, x, y, x_val, y_val, config, save_dir):
        ensure_dir(save_dir)
        file_path = join(save_dir, 'VanillaGRU_best_val.keras')
        cp_callback = ModelCheckpoint(filepath=file_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      mode='max', verbose=0)

        history = self.model.fit(x, y, epochs=config['epochs'], batch_size=config['batch_size'],
                                 validation_data=(x_val, y_val), shuffle=True,
                                 callbacks=[cp_callback])

        return history
    

class ConvMixGRU(SequentialModel):
    def __init__(self):
        super(ConvMixGRU, self).__init__()

    def build_model(self, config):
        input_shape = config['input_shape']
        model = self.model
        model.add(Conv1D(22, 10,
                         input_shape=(input_shape[0], input_shape[1]),
                         kernel_regularizer=l2(config['l2'])))
        model.add(BatchNormalization(axis=1))
        model.add(MaxPooling1D(2))
        if config['LSTM']:
            model.add(LSTM(44, kernel_regularizer=l2(config['l2']), return_sequences=True))
        else:
            model.add(GRU(44, kernel_regularizer=l2(config['l2']), return_sequences=True))
        model.add(Dropout(config['dropout']))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dropout(config['dropout']))
        model.add(Dense(4, activation='softmax'))
        optimizer = Adam(learning_rate=config['lr'], beta_1=0.9, beta_2=0.999,
                         amsgrad=False)
        model.compile(optimizer=optimizer,
                      loss=CategoricalCrossentropy(),
                      metrics=['accuracy'])
        model.summary()
        print("Model compiled.")

    def train(self, x, y, x_val, y_val, config, save_dir):
        ensure_dir(save_dir)
        file_path = join(save_dir, 'ConvMixGRU_best_val.keras')
        cp_callback = ModelCheckpoint(filepath=file_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      mode='max', verbose=0)

        history = self.model.fit(x, y, epochs=config['epochs'],
                                 batch_size=config['batch_size'],
                                 validation_data=(x_val, y_val), shuffle=True,
                                 callbacks=[cp_callback], verbose=1)
        return history


class AvgPoolCNN(SequentialModel):
    def __init__(self):
        super(AvgPoolCNN, self).__init__()

    def build_model(self, config):

        inputs = Input(shape=(config["input_shape"][0], config["input_shape"][1]))

        layer = inputs

        layer = Reshape((config["input_shape"][0], config["input_shape"][1], 1))(layer)

        layer = Conv2D(48, (1, 10), activation='elu',
                       kernel_regularizer=l2(config['l2']))(layer)
        layer = BatchNormalization()(layer)
        # layer = Conv2D(15, (1, 15), activation='elu',kernel_regularizer=regularizers.l2(0.02))(layer)
        layer = Dropout(0.1)(layer)
        layer = Conv2D(40, (22, 1), activation='elu',
                       kernel_regularizer=l2(config['l2']))(layer)
        layer = BatchNormalization()(layer)
        # layer = Conv2D(14, (22, 1), activation='elu')(layer)
        layer = AveragePooling2D((1, 25), strides=(1, 4))(layer)

        layer = Flatten()(layer)

        layer = Dense(4)(layer)
        outputs = Activation('softmax')(layer)
        self.model = Model(inputs=inputs, outputs=outputs)

        optimizer = Adam(learning_rate=config['lr'], beta_1=0.9, beta_2=0.999,
                         amsgrad=False)
        self.model.compile(optimizer=optimizer,
                      loss=CategoricalCrossentropy(),
                      metrics=['accuracy'])
        self.model.summary()
        print("Model compiled.")

    def train(self, x, y, x_val, y_val, config, save_dir):
        ensure_dir(save_dir)
        file_path = join(save_dir, 'AvgPoolCNN_best_val.keras')
        cp_callback = ModelCheckpoint(filepath=file_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      mode='max', verbose=0)

        history = self.model.fit(x, y, epochs=config['epochs'],
                                 batch_size=config['batch_size'],
                                 validation_data=(x_val, y_val), shuffle=True,
                                 callbacks=[cp_callback])
        return history


class threeLayerCNN(SequentialModel):
    def __init__(self):
        super(threeLayerCNN, self).__init__()
        
    def build_model(self, config):
        model = self.model
        
        input_shape = config["input_shape"]
        lr = config.get('lr', 0.001)
        decay = config.get("decay", 0.01)
 
        model.add(Conv2D(25, kernel_size=(3, 1), strides=1, padding='valid', kernel_regularizer=L1L2(l1=0, l2=0.01), input_shape=input_shape, activation='elu', data_format='channels_last'))
        model.add(BatchNormalization(axis=1))
        model.add(MaxPooling2D(pool_size=(3,1), padding='same')) # Read the keras documentation
        model.add(Dropout(0.5))

        model.add(Conv2D(32, kernel_size=(7,1), strides=1, padding='valid', kernel_regularizer=L1L2(l1=0, l2=0.01), activation='elu'))
        model.add(BatchNormalization(axis=1))
        model.add(MaxPooling2D(pool_size=(1,2), padding='same'))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, kernel_size=(5,1), strides=1, padding='valid', kernel_regularizer=L1L2(l1=0, l2=0.01), activation='elu'))
        model.add(BatchNormalization(axis=1))
        model.add(MaxPooling2D(pool_size=(1,2), padding='same'))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(16))
        model.add(BatchNormalization(axis=1))
        model.add(Dropout(0.5))
        model.add(Dense(4, kernel_regularizer=L1L2(l1=0, l2=0.01), activation='softmax'))
        optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999,
                         amsgrad=False, epsilon=1e-8, decay=decay)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.summary()
        print("Model compiled.")
        
    def train(self, x, y, x_val, y_val, config, save_dir):
        ensure_dir(save_dir)
        file_path = join(save_dir, 'threeLayerCNN.keras')
        cp_callback = ModelCheckpoint(filepath=file_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      mode='max', verbose=0)

        history = self.model.fit(x, y, epochs=config['epochs'], batch_size=config['batch_size'],
                                 validation_data=(x_val, y_val), shuffle=True,
                                 callbacks=[cp_callback])

        return history

    
class CNN4LayerGRU(SequentialModel):
    def __init__(self):
        super(CNN4LayerGRU, self).__init__()
        
    def build_model(self, config):
        model = self.model
        
        input_shape = config["input_shape"]
        lr = config.get('lr', 0.001)
        decay = config.get("decay", 0.01)
 
        # conv. block 1
        model.add(Conv2D(filters=25, kernel_size=(5,5), padding='same', activation='elu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(3,1), padding='same')) # Read the keras documentation
        model.add(BatchNormalization())
        model.add(Dropout(0.6))

        #block 2
        model.add(Conv2D(filters=50, kernel_size=(5,5), padding='same', activation='elu'))
        model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.6))

        #block 3
        model.add(Conv2D(filters=100, kernel_size=(5,5), padding='same', activation='elu'))
        model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.6))

        #block4
        model.add(Conv2D(filters=200, kernel_size=(5,5), padding='same', activation='elu'))
        model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.6))

        model.add(Flatten())
    
        model.add(Reshape((200, 4))) 
        model.add(GRU(44, kernel_regularizer=L1L2(l1=0, l2=0.1), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dropout(0.2))

        # Output layer with Softmax activation 
        model.add(Dense(4, activation='softmax')) # Output FC layer with softmax activation
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
        
        model.summary()
        print("Model compiled.")
        
    def train(self, x, y, x_val, y_val, config, save_dir):
        ensure_dir(save_dir)
        file_path = join(save_dir, 'CNN4LayerGRU.keras')
        cp_callback = ModelCheckpoint(filepath=file_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      mode='max', verbose=0)

        history = self.model.fit(x, y, epochs=config['epochs'], batch_size=config['batch_size'],
                                 validation_data=(x_val, y_val), shuffle=True,
                                 callbacks=[cp_callback])

        return history
    

class CNN4LayerLSTM(SequentialModel):
    def __init__(self):
        super(CNN4LayerLSTM, self).__init__()
        
    def build_model(self, config):
        model = self.model
        
        input_shape = config["input_shape"]
        lr = config.get('lr', 0.001)
        decay = config.get("decay", 0.01)
 
         # Conv. block 1
        model.add(Conv2D(filters=25, kernel_size=(5,5), padding='same', activation='elu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(3,1), padding='same')) # Read the keras documentation
        model.add(BatchNormalization())
        model.add(Dropout(0.6))

        # Conv. block 2
        model.add(Conv2D(filters=50, kernel_size=(5,5), padding='same', activation='elu'))
        model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.6))

        # Conv. block 3
        model.add(Conv2D(filters=100, kernel_size=(5,5), padding='same', activation='elu'))
        model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.6))

        # Conv. block 4
        model.add(Conv2D(filters=200, kernel_size=(5,5), padding='same', activation='elu'))
        model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.6))

        # FC+LSTM layers
        model.add(Flatten()) # Adding a flattening operation to the output of CNN block
        model.add(Dense((40))) # FC layer with 100 units
        model.add(Reshape((40,1))) # Reshape my output of FC layer so that it's compatible
        model.add(LSTM(10, dropout=0.4, recurrent_dropout=0.1, input_shape=(40,1), return_sequences=False))


        # Output layer with Softmax activation 
        model.add(Dense(4, activation='softmax')) # Output FC layer with softmax activation
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
        
        model.summary()
        print("Model compiled.")
        
    def train(self, x, y, x_val, y_val, config, save_dir):
        ensure_dir(save_dir)
        file_path = join(save_dir, 'CNN4LayerGRU.keras')
        cp_callback = ModelCheckpoint(filepath=file_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      mode='max', verbose=0)

        history = self.model.fit(x, y, epochs=config['epochs'], batch_size=config['batch_size'],
                                 validation_data=(x_val, y_val), shuffle=True,
                                 callbacks=[cp_callback])

        return history
    

class CNN2LayerLSTM(SequentialModel):
    def __init__(self):
        super(CNN2LayerLSTM, self).__init__()
        
    def build_model(self, config):
        model = self.model
        
        input_shape = config["input_shape"]
        lr = config.get('lr', 0.001)
        decay = config.get("decay", 0.01)
 
        model.add(Conv2D(25, kernel_size=(3, 1), strides=1, padding='valid', kernel_regularizer=L1L2(l1=0, l2=0.01), input_shape=input_shape, activation='elu', data_format='channels_last'))
        model.add(BatchNormalization(axis=1))
        model.add(MaxPooling2D(pool_size=(3,1), padding='same')) # Read the keras documentation
        model.add(Dropout(0.5))

        model.add(Conv2D(32, kernel_size=(7,1), strides=1, padding='valid', kernel_regularizer=L1L2(l1=0, l2=0.01), activation='elu'))
        model.add(BatchNormalization(axis=1))
        model.add(MaxPooling2D(pool_size=(1,2), padding='same'))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, kernel_size=(5,1), strides=1, padding='valid', kernel_regularizer=L1L2(l1=0, l2=0.01), activation='elu'))
        model.add(BatchNormalization(axis=1))
        model.add(MaxPooling2D(pool_size=(1,2), padding='same'))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(40))
        model.add(BatchNormalization(axis=1))
        model.add(Dropout(0.5))

        # FC+LSTM layers
        model.add(Reshape((40,1))) # Reshape my output of FC layer so that it's compatible
        model.add(LSTM(16, dropout=0.4, recurrent_dropout=0.1, return_sequences=True))
        model.add(LSTM(10, dropout=0.4, recurrent_dropout=0.1, return_sequences=False))

        # Output layer with Softmax activation 
        model.add(Dense(4, activation='softmax')) # Output FC layer with softmax activation
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
        
        model.summary()
        print("Model compiled.")
        
    def train(self, x, y, x_val, y_val, config, save_dir):
        ensure_dir(save_dir)
        file_path = join(save_dir, 'CNN2LayerLSTM.keras')
        cp_callback = ModelCheckpoint(filepath=file_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      mode='max', verbose=0)

        history = self.model.fit(x, y, epochs=config['epochs'], batch_size=config['batch_size'],
                                 validation_data=(x_val, y_val), shuffle=True,
                                 callbacks=[cp_callback])

        return history
    

class CNN2LayerGRU(SequentialModel):
    def __init__(self):
        super(CNN2LayerGRU, self).__init__()
        
    def build_model(self, config):
        model = self.model
        
        input_shape = config["input_shape"]
        lr = config.get('lr', 0.001)
        decay = config.get("decay", 0.01)
 
        # Conv. block 1
        model.add(Conv2D(filters=25, kernel_size=(5,5), padding='same', activation='elu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(3,1), padding='same')) # Read the keras documentation
        model.add(BatchNormalization())
        model.add(Dropout(0.6))

        # Conv. block 2
        model.add(Conv2D(filters=50, kernel_size=(5,5), padding='same', activation='elu'))
        model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.6))

        # Conv. block 3
        model.add(Conv2D(filters=100, kernel_size=(5,5), padding='same', activation='elu'))
        model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.6))

        # Conv. block 4
        model.add(Conv2D(filters=200, kernel_size=(5,5), padding='same', activation='elu'))
        model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.6))

        # FC+LSTM layers
        model.add(Flatten()) # Adding a flattening operation to the output of CNN block
        model.add(Dense((40))) # FC layer with 100 units
        model.add(Reshape((40,1))) # Reshape my output of FC layer so that it's compatible
        model.add(GRU(64, dropout=0.4, recurrent_dropout=0.1, input_shape=(40,1), return_sequences=True))
        model.add(GRU(16, dropout=0.4, recurrent_dropout=0.1, return_sequences=False))


        # Output layer with Softmax activation 
        model.add(Dense(4, activation='softmax')) # Output FC layer with softmax activation

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        
        model.summary()
        print("Model compiled.")
        
    def train(self, x, y, x_val, y_val, config, save_dir):
        ensure_dir(save_dir)
        file_path = join(save_dir, 'CNN2LayerGRU.keras')
        cp_callback = ModelCheckpoint(filepath=file_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      mode='max', verbose=0)

        history = self.model.fit(x, y, epochs=config['epochs'], batch_size=config['batch_size'],
                                 validation_data=(x_val, y_val), shuffle=True,
                                 callbacks=[cp_callback])

        return history