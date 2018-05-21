# --> library from keras
# --> library from tensorflow
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# --> library from sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from sklearn.ensemble import AdaBoostRegressor

# --> library from matplotlib
from matplotlib import pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

class SurrogateML:
    """
    Class implementation of surrogate model
    """
    def __init__(self, name_of_model, hyp_config):
        """Init for surrogate machine learning model

        Parameters
        -----------
        name_of_model : string
            The name of the model to choose
        hyp_config : dictionary
            The configuration directionary for hyperparameters
            if 'ann' == name_of_model:
            'layer_struct'
            'activation_function'
            'verbose'
            'dropout'
            'scaling'
            'loss'
            'npoch'
            'batch_size'
            'lr'

        """
        self.name_of_model = name_of_model

        if self.name_of_model == 'ann':
            self.build_ann(hyp_config)

        if self.name_of_model == 'adaboost':
            self.build_adb(hyp_config)

    def build_adb(self, hyp_config):

        n_estimator = hyp_config['n_estimator']
        learning_rate = hyp_config['lr']
        loss_type = hyp_config['loss_type']


        self.model = AdaBoostRegressor(n_estimators=n_estimator,
                                       learning_rate=learning_rate,
                                       loss=loss_type)


    def build_ann(self, hyp_config):

        self.model = Sequential()

        # --> set layer structure
        layer_struct = hyp_config['layer_struct']

        # --> set activation function
        if hyp_config.get('activation_function') == None:
            act_fun = 'relu'
        else:
            act_fun = hyp_config['activation_function']

        # --> set verbose
        flag_verbose = hyp_config.get('verbose')

        # --> set dropout
        if hyp_config.get('dropout') == None:
            flag_dropout = False
        else:
            flag_dropout = True
            dropout_rate = hyp_config.get('dropout')

        # --> set scaling
        if hyp_config.get('scaling') == None:
            self.scaling = True
        else:
            self.scaling = hyp_config['scaling']

        # --> set loss
        if hyp_config.get('loss') == None:
            loss = 'kullback_leibler_divergence'
        else:
            loss = hyp_config['loss']

        # --> set npoch
        if hyp_config.get('npoch') == None:
            self.npoch = 1000
        else:
            self.npoch = hyp_config['npoch']

        # --> set batch size
        if hyp_config.get('batch_size') == None:
            self.bs = 32
        else:
            self.bs = hyp_config['batch_size']

        # --> set learning rate
        if hyp_config.get('lr') == None:
            self.lr = 1e-3
        else:
            self.lr = hyp_config.get('lr')

        # inner nonlinear
        for index_unit in xrange(len(layer_struct)-1):
            self.model.add(Dense(layer_struct[index_unit+1], input_dim=layer_struct[index_unit]))
            self.model.add(Activation(act_fun))
            if flag_dropout: self.model.add(Dropout(dropout_rate))

        # last linear
        self.model.add(Dense(1, input_dim=layer_struct[-1]))

        # compile model
        self.model.compile(loss=loss, optimizer=Adam(lr=self.lr ))

        # output structure of neural network
        self.model.summary()
        if flag_verbose:
            self.train_verbose = 1
        else:
            self.train_verbose = 0

    def train_on_data(self, feature, target):

        self.num_train_samples = feature.shape[0]

        if self.scaling:
            self.scaler_feature = StandardScaler()
            self.scaler_target = StandardScaler()
            feature_ts = self.scaler_feature.fit_transform(feature)
            target_ts = self.scaler_target.fit_transform(target)
        else:
            feature_ts = feature
            target_ts = target

        # early stopping in NN
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=self.npoch, \
                                  verbose=1, mode='auto')
        callbacks_list = [earlystop]

        self.history = self.model.fit(feature_ts, target_ts,
                                      epochs=self.npoch,
                                      batch_size=self.bs,
                                      validation_split=0.1,
                                      shuffle=True,
                                      verbose=self.train_verbose,
                                      callbacks=callbacks_list
                                      ).history


    # def transform_to_prob(self, prob):
    #     return np.exp(prob)

    def get_history(self):
        return self.history

    def plot_lr(self):
        # pass
        plt.figure()

        train_loss = self.history['loss']
        val_loss = self.history['val_loss']
        plt.semilogy(train_loss, 'r-', label='training loss')
        plt.semilogy(val_loss, 'b-' ,label='validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title('learning curve')
        plt.legend(['training loss', 'validation loss'])
        plt.savefig('lr_' + str(self.num_train_samples) + '.png')

        plt.close()

    def plot_r2_xy_scatter(self, true, pred):

        # compute r2
        r2 = r2_score(true, pred)

        y_min = min(true)
        y_max = max(true)

        plt.figure()
        plt.scatter(pred, true, c='k', s=20)
        plt.plot(true, true, '-r')
        plt.xlabel('prediction')
        plt.ylabel('truth')
        plt.xlim([y_min, y_max])
        plt.ylim([y_min, y_max])
        plt.title('performance on testing data with $R^2$ score: '+str(r2))
        plt.savefig('xy_scatter_'+str(self.num_train_samples)+'.png')
        plt.close()


    def compute_mse_prediction_error(self, feature, true):

        pred = self.predict(feature)

        mse_error = mean_squared_error(true, pred)

        self.plot_r2_xy_scatter(true, pred)

        return mse_error


    def predict(self, feature):

        if self.scaling:
            feature_ts = self.scaler_feature.fit_transform(feature)
        else:
            feature_ts = feature

        pred_ts = self.model.predict(feature_ts)

        if self.scaling:
            pred = self.scaler_target.inverse_transform(pred_ts)
        else:
            pred = pred_ts

        return pred
