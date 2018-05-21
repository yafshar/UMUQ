import os
import sys
import numpy as np
import time
import copy

from plot_target_scatter import plot_target_scatter
from ClassSurrogate import SurrogateML
from data_generator import yaser_4_val_1_f_data_generator
from matplotlib import pyplot as plt

# timer start
start = time.time()

######################################################################

cut_off = -3
col_names = ['1f','2f','3f','4f','funcValue','ignore' ]
feature_col = col_names[:-2]
target_col = col_names[-2]

def pd_feature_target_decompose(df):
    feature = np.array(df[feature_col])
    target = np.array(df[target_col])
    target = target.reshape(-1,1)
    return feature, target

dir_data = './reml/'

# file directory
file_list = os.listdir('./reml')
file_list.sort()
num_file = len(file_list)

# data generator
data_generator = yaser_4_val_1_f_data_generator(file_list=file_list,
                                                dir_data=dir_data,
                                                transform=True,
                                                col_names=col_names,
                                                target_col=target_col,
                                                log_cut_off=cut_off)

# ann model
hyp_config = {'model': 'ann',
             'layer_struct':[4,20,20],
             'activation_function':'selu',
             'verbose':False,
             'scaling':True,
             'loss':'mse',
             'npoch':20000,
             'batch_size':256,
             'lr':5e-4}


mse_error_on_test = []

# ## parallel mode
# # input number of training data used
# number_train_data = int(sys.argv[1])
#
# for index_test in range(number_train_data - 1):
#     data_generator.add_next_set()
#
#
# training_data_frame = data_generator.get_current_set()
# print 'training on: ', file_list[:number_train_data]
# print 'current number of training points: ', training_data_frame.shape[0]
# train_feature, train_target = pd_feature_target_decompose(training_data_frame)
#
# # refresh model for each training data
# surrogate_model = SurrogateML(name_of_model=hyp_config['model'], hyp_config=hyp_config)
#
# # training on current data
# surrogate_model.train_on_data(train_feature, train_target)
# surrogate_model.plot_lr()
# print ''
# print '....finished training....'
#
# # obtain next data sets
# testing_data_frame = data_generator.get_next_set()
# print 'current number of testing points: ', testing_data_frame.shape[0]
# test_feature, test_target = pd_feature_target_decompose(testing_data_frame)
#
#
#
#
# # evaluate prediction error on next data sets
# mse_on_test = surrogate_model.compute_mse_prediction_error(test_feature, test_target)
#
# print 'number of train data: ', str(number_train_data), ' mean squared error on test', mse_on_test
#
# # obtain prediction
# test_pred = surrogate_model.predict(test_feature)
#
# # obtain a df for prediction
# test_pred_df = copy.deepcopy(testing_data_frame)
# # test_pred_df = testing_data_frame.copy()
# del test_pred_df[target_col]
# test_pred_df[target_col] = test_pred
#
# # debug
# print testing_data_frame.describe()
# print test_pred_df.describe()
#
# # save df to csv
# testing_data_frame.to_csv('df_true_' + str(number_train_data) + '.csv')
# test_pred_df.to_csv('df_model_' + str(number_train_data) + '.csv')
#
# # plot scatter for testing data
# plot_target_scatter(number_train_data, testing_data_frame, cut_off, 'true')
# plot_target_scatter(number_train_data, test_pred_df, cut_off, 'model')


## sequential mode
for index_test in xrange(1, num_file):  # num_file

    # refresh model for each training data
    surrogate_model = SurrogateML(name_of_model='ann', hyp_config=hyp_config)
    #
    print '====================================================='
    print 'current training set list: ', file_list[:index_test]

    # obtain current data sets: accumulated data sets upto index_test-th data
    training_data_frame = data_generator.get_current_set()
    print 'current number of training points: ', training_data_frame.shape[0]
    train_feature, train_target = pd_feature_target_decompose(training_data_frame)

    # training on current data
    surrogate_model.train_on_data(train_feature, train_target)
    surrogate_model.plot_lr()
    print ''
    print '....finished training....'

    # obtain next data sets
    testing_data_frame = data_generator.get_next_set()
    print 'current number of testing points: ', testing_data_frame.shape[0]
    test_feature, test_target = pd_feature_target_decompose(testing_data_frame)

    # evaluate prediction error on next data sets
    mse_on_test = surrogate_model.compute_mse_prediction_error(test_feature, test_target)
    mse_error_on_test.append(mse_on_test)

    # update data generator to next
    data_generator.add_next_set()

    print 'mse on test = ', mse_on_test
    print '====================================================='




print '====================================================='
print 'Error tendency of surrogate model'
plt.figure()
plt.plot(mse_error_on_test)
plt.xlabel('number of generation')
plt.ylabel('loss on next generation')
plt.savefig('error_vs_generation.png')
plt.close()


# timer end
end = time.time()

elapsed = end - start

print ''
print '============================='
print ' total computational time = ', elapsed, ' sec '
