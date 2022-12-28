# import necessary packages and modules, available packages are listed in the following:
# numpy, scipy, mne

# user-defined custom function
# - feed_data_func: function that takes the data from GUI and returns the data to be plotted
import os
import pickle
import joblib
import sys
import platform

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import mne
from scipy import signal
from pathlib import Path
import mlflow


class Custom:

    def _check_params(self):
        # check if params is a dict and has the required keys
        if not isinstance(self.params, dict) or not all(
                key in self.params for key in ('graph_type', 'buffer_size', 'update_interval')):
            raise ValueError('params must be a dict and have the required keys')

    def __init__(self):

        # Params for graphs seeting
        # required keys: graph_type , update_interval
        self.params = dict(
            # 'line' or 'bar', required
            graph_type='bar',
            # in ms, required
            update_interval=100,
            # in seconds, required
            buffer_size=5,
            # optional, but highly recommend to set
            # x_range=[0, 2500],
            # optional, but highly recommend to set
            # y_range=[-1e-4, 1e4],
            # optional, but highly recommend to set
            # x_ticks=[0, 500, 1000, 1500, 2000, 2500],
            # optional, but highly recommend to set
            # y_ticks=[0, 500, 1000, 1500, 2000, 2500],
            # optional, but highly recommend to set
            x_label='Poor vs Normal',
            # optional, but highly recommend to set
            y_label='Probability',
            # optional, but highly recommend to set
            title='Signal Quality',
        )

        self._check_params()

        self.feat_list = [
            'activity', 'mobility', 'complexity',
            'mean', "var",'std', "ptp", "minim", "maxim", "rms", "abs_diff",
            'skewness', 'kurtosis', "deg1", "deg2",
        ]

        # macOS
        if platform.system() == 'Darwin':
            server_dir = Path('/Volumes/mindampshared/')
        # Windows
        elif platform.system() == 'Windows':
            server_dir = Path('Z:')
        # Linux
        elif platform.system() == 'Linux':
            # TODO: add linux path
            server_dir = Path('/mnt/mindampshared/')
        else:
            raise ValueError('Unsupported OS')

        self.model_dir = "Signal_QA_model/best_model_2classes_nowData"
        self.cls_model = mlflow.pyfunc.load_model(self.model_dir)

    def Hjorth_params(self, data):

        def activity(x):
            return np.var(x)

        def mobility(x):
            x_first = np.diff(x, 1)
            ratio = activity(x_first) / activity(x)
            return np.sqrt(ratio)

        def complexity(x):
            x_first = np.diff(x, 1)
            return mobility(x_first) / mobility(x)

        feat_dict = {
            'activity': activity(data),
            'mobility': mobility(data),
            'complexity': complexity(data),
        }

        return feat_dict

    def statistical_features(self,data):
        def mean(x):
            return np.mean(x)

        def var(x):
            return np.var(x)

        def std(x):
            return np.std(x)

        def ptp(x):
            return np.ptp(x)

        def minim(x):
            return np.min(x)

        def maxim(x):
            return np.max(x)

        def rms(x):
            return np.sqrt(np.mean(np.power(x, 2)))

        def abs_diff(x):
            return np.sum(np.abs(np.diff(x, 1)))

        def skewness(x):
            return stats.skew(x)

        def kurtosis(x):
            return stats.kurtosis(x)

        return {
            "mean": mean(data),
            "var": var(data),
            "std": std(data),
            "ptp": ptp(data),
            "minim": minim(data),
            "maxim": maxim(data),
            "rms": rms(data),
            "abs_diff": abs_diff(data),
            "skewness": skewness(data),
            "kurtosis": kurtosis(data),
        }

    def poly_norm_features(self, data):
        deg1 = []
        deg2 = []
        x = np.arange(0, len(data), 1)
        coef = np.polyfit(x=x, y=data, deg=3)
        deg1.append(coef[-1])
        deg2.append(coef[-2])
        return {"deg1": np.array(deg1),
                "deg2": np.array(deg2),
                }

    def qa_filtering(self, signal_data, low_cut, high_cut, notch_value=50, fs=500.5):
        filtered_signal = signal_data.copy()

        # notch filtering
        for value in np.arange(notch_value, 250, notch_value):
            sos = signal.butter(N=2, Wn=[value - 2, value + 2], btype='bandstop', fs=fs, output='sos')
            filtered_data = signal.sosfilt(sos, filtered_signal)

        # bandpass filtering
        high_pass_sos = signal.butter(N=2, Wn=low_cut, btype='highpass', fs=fs, output='sos')
        filtered_data = signal.sosfilt(high_pass_sos, filtered_data)

        low_pass_sos = signal.butter(N=2, Wn=high_cut, btype='lowpass', fs=fs, output='sos')
        filtered_data = signal.sosfilt(low_pass_sos, filtered_data)

        return filtered_data

    # def band_rms(self, signal_data):
    #     # get band signal
    #     alpha = self.qa_filtering(signal_data, low_cut=8, high_cut=12)
    #     beta = self.qa_filtering(signal_data, low_cut=12, high_cut=30)
    #     non_alpha = signal_data - alpha
    #     non_beta = signal_data - beta
    #
    #     def rms(x):
    #         return np.sqrt(np.mean(np.power(x, 2)))
    #
    #     feat_dict = {
    #         "alpha_rms": rms(alpha),
    #         "beta_rms": rms(beta),
    #         "non_alpha_rms": rms(non_alpha),
    #         "non_beta_rms": rms(non_beta),
    #     }

        # return feat_dict

    def feature_extraction(self, signal_data):
        feature_dict = dict()
        # extract different types of features
        for feat_func in [self.Hjorth_params, self.statistical_features, self.poly_norm_features]:
            feat_dict_new = feat_func(signal_data)
            feature_dict.update(feat_dict_new)

        df_feature = pd.DataFrame.from_dict([feature_dict])
        df_feature = df_feature[self.feat_list]
        features = df_feature.values
        return features

    def qa_preprocessing(self, signal_data, input_duration=1, fs=500.5):
        """
        :param signal_data: np.array, 1D, shape(n,), n is the number of samples. in micro volt
        :param input_duration: float, duration of input signal, in second, model input length
        :param fs: float, sampling frequency, in Hz
        :return features: np.array, 1D, shape(1, -1), features of input signal
        """
        # filtering
        filtered_data = self.qa_filtering(signal_data, low_cut=1, high_cut=45, fs=fs)

        # scale to volt unit
        filtered_data_inV = filtered_data / 1e6

        # crop filtered data
        filtered_data_crop = filtered_data_inV[-int(input_duration * fs):].copy()

        return filtered_data_crop

    def signal_quality_prediction(self, features, model):
        """
        :param features: features from EEG signal
        :param model: trained model
        :return: probs (probabilty of Poor, Normal)
        """
        # extract feature
        # probs = model.predict_proba(features)
        probs = model.predict(features)
        if probs == 0:
            label = np.array([1, 0])
        else:
            label =np.array([0, 1])
        return label

    def custom_func(self, data):
        """
        :param data: (1d numpy.array, or list), shape: (n, ), data feeding from GUI, n is the number of samples
        :return: (1d numpy.array), shape: (n, ), n is the number of dimension of the output (normal vs poor)
        """
        # preprocessing
        filtered_data_crop = self.qa_preprocessing(data)

        # features
        features = self.feature_extraction(filtered_data_crop)
        features = np.array(features,dtype="float64")
        features.reshape(1, -1)
        # make prediction
        probs = self.signal_quality_prediction(features, self.cls_model)

        return probs
