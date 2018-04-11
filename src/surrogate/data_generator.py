import pandas as pd
import numpy as np

class yaser_4_val_1_f_data_generator:

    def __init__(self, file_list, dir_data, transform, col_names, target_col, log_cut_off):
        self.num_current_set = 0
        self.file_list = file_list
        self.dir_data = dir_data
        self.transform = transform
        self.target_col = target_col
        self.col_names = col_names
        self.cut_off = log_cut_off

        # first data into df
        path_data = self.dir_data + self.file_list[self.num_current_set]
        self.df = pd.read_csv(path_data, header=None, delim_whitespace=True)
        self.df.columns = col_names

        # transform target to real prob if requested
        if self.transform: self.transform_target(self.df)

        # remove ignore column
        del self.df['ignore']

        # update file index
        self.num_current_set = self.num_current_set + 1

    def shaowu_cut(self, x):
        if x <= self.cut_off:
            return self.cut_off
        else:
            return x

    def transform_target(self, data_frame):

        # 1. transform to real space..
        # data_frame[self.target_col] = data_frame[self.target_col].apply(np.exp)

        # 2. cut log(prob) below a certain threshold to the certain threshold
        data_frame[self.target_col] = data_frame[self.target_col].apply(self.shaowu_cut)

    def get_next_set(self):
        path_data = self.dir_data + self.file_list[self.num_current_set]
        df_input = pd.read_csv(path_data, header=None, delim_whitespace=True)
        df_input.columns = self.col_names
        del df_input['ignore']

        # transform target to real prob if requested
        if self.transform: self.transform_target(df_input)

        return df_input

    def add_next_set(self):
        # get next set
        df_input = self.get_next_set()

        print 'next set information'
        print df_input.describe()

        # append truncated_df_input into df
        self.df = self.df.append(df_input, ignore_index=True)
        self.num_current_set = self.num_current_set + 1

    def get_current_set(self):
        return self.df

    def get_summary(self):
        print self.df.describe()