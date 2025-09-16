import os

import joblib
import torch
import pandas as pd
import numpy as np

from prediction.MLPModel import MLPModel
import prediction.ModelOutputFileName as ModelOutputFileName
from prediction.ModelType import ModelType


class MLPModelManager:

    def __init__(self, model_dir_path: str, model_type: ModelType, with_server_usage_limit_input: bool):
        self._model_dir_path = model_dir_path
        self._model_type = model_type
        self._with_server_usage_limit_input = with_server_usage_limit_input

        self._input_size = None
        self._output_size = None
        self._model = None
        self._input_url_encoder = None
        self._input_log_scaler = None
        self._input_other_scaler = None
        self._output_scaler = None

    def predict(self, input_url_df: pd.DataFrame, input_log_df: pd.DataFrame | None, input_other_df: pd.DataFrame):
        self._load_model(input_log_df is not None)

        # preprocess
        input_url_encoded = self._input_url_encoder.transform(input_url_df)
        if self._input_log_scaler is not None:
            input_log_scaled = self._input_log_scaler.transform(input_log_df)
        else:
            input_log_scaled = None
        input_other_scaled = self._input_other_scaler.transform(input_other_df)

        if input_log_scaled is not None:
            input_processed = np.concatenate([input_url_encoded, input_log_scaled, input_other_scaled], axis=1)
        else:
            input_processed = np.concatenate([input_url_encoded, input_other_scaled], axis=1)

        # predict
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_processed)
            output = self._model(input_tensor).numpy()

        prediction = self._output_scaler.inverse_transform(output)
        print(f'---------- prediction input (input size: {self._input_size}) (origin) ----------')
        print('----------- url')
        print(input_url_df.values)
        if input_log_df is not None:
            print('----------- log')
            print(input_log_df.values)
        print('----------- other')
        print(input_other_df.values)

        print(f'---------- prediction input (input size: {self._input_size}) (processed) ----------')
        for i in range(self._input_size):
            print(f'{i}:', input_processed[0][i])

        print(f'---------- prediction result (output size: {self._output_size}) ----------')
        for i in range(self._output_size):
            print(f'{i}:', prediction[0][i])

        return prediction

    def _load_model(self, use_input_log_scaler: bool):
        print('---------- load model info ----------')
        print(self._model_dir_path)
        print(f'{ModelOutputFileName.get_model_name(self._model_type, self._with_server_usage_limit_input)}')

        input_url_encoder_file = ModelOutputFileName.get_input_url_encoder_file_name(self._model_type,
                                                                                     self._with_server_usage_limit_input)
        self._input_url_encoder = joblib.load(os.path.join(self._model_dir_path, input_url_encoder_file))

        if use_input_log_scaler:
            input_log_scaler_file = ModelOutputFileName.get_input_log_scaler_file_name(self._model_type,
                                                                                       self._with_server_usage_limit_input)
            self._input_log_scaler = joblib.load(os.path.join(self._model_dir_path, input_log_scaler_file))

        input_other_scaler_file = ModelOutputFileName.get_input_other_scaler_file_name(self._model_type,
                                                                                       self._with_server_usage_limit_input)
        self._input_other_scaler = joblib.load(os.path.join(self._model_dir_path, input_other_scaler_file))

        output_scaler_file = ModelOutputFileName.get_output_scaler_file_name(self._model_type, self._with_server_usage_limit_input)
        self._output_scaler = joblib.load(os.path.join(self._model_dir_path, output_scaler_file))

        model_params_file = ModelOutputFileName.get_model_params_file_name(self._model_type, self._with_server_usage_limit_input)
        model_params = torch.load(os.path.join(self._model_dir_path, model_params_file))
        # get input_size and output size from model params
        self._input_size = model_params[next(iter(model_params))].shape[1]
        self._output_size = model_params[next(reversed(model_params))].shape[0]
        self._model = MLPModel(self._input_size, self._output_size)
        self._model.load_state_dict(model_params)
        self._model.eval()
