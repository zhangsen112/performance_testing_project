import os
import itertools
import logging

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prediction.ModelData import ModelData
from prediction.ModelType import ModelType
from prediction.MLPModel import MLPModel
import prediction.ModelOutputFileName as ModelOutputFileName

TRAIN_PARAM_EPOCH = 100
TRAIN_PARAM_BATCH_SIZE = 32
TRAIN_PARAM_LEARNING_RATE = 1e-5
TRAIN_PARAM_RANDOM_STATE = 42
TRAIN_PARAM_TOLERANCE = 0.15
TRAIN_PARAM_TRAIN_SIZE = 0.6


class TrainManager:
    def __init__(self, dataset_file_path: str, output_dir_path: str, logger: logging.Logger | None = None):
        self._data_df = pd.read_csv(dataset_file_path)
        self._output_dir_path = output_dir_path
        self._logger = logger

    def train_app_capacity_models(self) -> float | None:
        model_types = {
            ModelType.APP_CAPACITY_HTTP,
            ModelType.APP_CAPACITY_SQL,
            ModelType.APP_CAPACITY_REDIS,
            ModelType.APP_CAPACITY_DNS,
        }
        return self._train_models(model_types)

    def train_server_capacity_models(self) -> float | None:
        model_types = {
            ModelType.SERVER_CAPACITY_HTTP,
            ModelType.SERVER_CAPACITY_SQL,
            ModelType.SERVER_CAPACITY_REDIS,
            ModelType.SERVER_CAPACITY_DNS,
            ModelType.SERVER_CAPACITY_QPS,
        }
        return self._train_models(model_types)

    def _train_models(self, model_type_set: set[ModelType]) -> float | None:
        model_type_list = [model_type for model_type in model_type_set]
        with_server_usage_limit_input_param_list = [True, False]

        model_acc_list = []
        for model_type, with_server_usage_limit_input in itertools.product(model_type_list,
                                                                           with_server_usage_limit_input_param_list):
            try:
                acc_dict = self._train_model(model_type, with_server_usage_limit_input)
                model_acc_list.append(np.mean([acc for acc in acc_dict.values()]))
            except Exception:
                return None

        model_acc = np.mean(model_acc_list)
        return model_acc

    def _train_model(self, model_type: ModelType, with_server_usage_limit_input: bool):
        model_data = ModelData(model_type, with_server_usage_limit_input, self._data_df)

        model_name = ModelOutputFileName.get_model_name(model_type, with_server_usage_limit_input)
        self._log(f'---------- {model_name}')
        # print(model_data.model_data_input_df.shape[0])
        # print(model_data.model_data_url_input_df)
        # print(model_data.model_data_log_input_df)
        # print(model_data.model_data_other_input_df)
        # print(model_data.model_data_output_df)

        # ▁▂▃▄ 设备设置 ▄▃▂▁
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 划分数据
        X = model_data.model_data_input_df
        y = model_data.model_data_output_df

        # ▁▂▃▄ 数据划分 ▄▃▂▁
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=1 - TRAIN_PARAM_TRAIN_SIZE, random_state=TRAIN_PARAM_RANDOM_STATE
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=TRAIN_PARAM_RANDOM_STATE
        )

        # ▁▂▃▄ 预处理定义 ▄▃▂▁
        # 动态识别特征类型
        log_features = [col for col in X.columns if "total" in col or "core" in col]
        url_features = ["URL"]
        numeric_features = [
            col for col in X.columns if col not in url_features and col not in log_features
        ]

        # 初始化预处理器
        url_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_log_scaler = StandardScaler()
        X_other_scaler = StandardScaler()
        y_scaler = StandardScaler()

        # ▁▂▃▄ 预处理拟合 ▄▃▂▁
        # 只在训练集上拟合
        url_encoder.fit(X_train[url_features])

        if log_features:
            X_log_scaler.fit(np.log(X_train[log_features]))

        X_other_scaler.fit(X_train[numeric_features])
        y_scaler.fit(y_train)

        # ▁▂▃▄ 预处理函数 ▄▃▂▁
        def preprocess_features(X_data):
            """动态特征预处理流水线"""
            processed_dfs = []

            # URL编码
            if url_features:
                url_encoded = url_encoder.transform(X_data[url_features])
                url_df = pd.DataFrame(
                    url_encoded, columns=url_encoder.get_feature_names_out(url_features)
                )
                processed_dfs.append(url_df)

            # 对数特征处理
            if log_features:
                log_transformed = np.log(X_data[log_features])
                log_scaled = X_log_scaler.transform(log_transformed)
                log_df = pd.DataFrame(
                    log_scaled, columns=[f"log_{col}" for col in log_features]
                )
                processed_dfs.append(log_df)

            # 其他数值特征
            if numeric_features:
                num_scaled = X_other_scaler.transform(X_data[numeric_features])
                num_df = pd.DataFrame(num_scaled, columns=numeric_features)
                processed_dfs.append(num_df)

            return pd.concat(processed_dfs, axis=1)

        # ▁▂▃▄ 应用预处理 ▄▃▂▁
        X_train_processed = preprocess_features(X_train)
        X_val_processed = preprocess_features(X_val)
        X_test_processed = preprocess_features(X_test)

        # 处理目标变量
        y_train_scaled = y_scaler.transform(y_train)
        y_val_scaled = y_scaler.transform(y_val)
        y_test_scaled = y_scaler.transform(y_test)

        # ▁▂▃▄ 转换为张量 ▄▃▂▁
        X_train_tensor = torch.FloatTensor(X_train_processed.values).to(device)
        y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
        X_val_tensor = torch.FloatTensor(X_val_processed.values).to(device)
        y_val_tensor = torch.FloatTensor(y_val_scaled).to(device)
        X_test_tensor = torch.FloatTensor(X_test_processed.values).to(device)
        y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)

        model = MLPModel(X_train_processed.shape[1], y_train_scaled.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=TRAIN_PARAM_LEARNING_RATE, weight_decay=1e-3)
        criterion = nn.MSELoss()

        # ▁▂▃▄ 训练循环 ▄▃▂▁
        # best_val_loss = float("inf")
        for epoch in range(TRAIN_PARAM_EPOCH):
            model.train()
            loss = None
            for i in range(0, len(X_train_tensor), TRAIN_PARAM_BATCH_SIZE):
                inputs = X_train_tensor[i: i + TRAIN_PARAM_BATCH_SIZE]
                labels = y_train_tensor[i: i + TRAIN_PARAM_BATCH_SIZE]

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # 验证损失计算
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)

            self._log(
                f"Epoch {epoch + 1:03d}"
                f" | Train Loss: {loss.item() if loss is not None else float('inf'):.4f}"
                f" | Val Loss: {val_loss.item():.4f}"
            )

        model.eval()
        with torch.no_grad():
            test_pred_normalized = model(X_test_tensor).cpu().numpy()

        test_pred = y_scaler.inverse_transform(test_pred_normalized)
        y_test_original = y_scaler.inverse_transform(y_test_tensor.cpu().numpy())

        # 计算各指标的准确率
        accuracy_dict = {}
        for idx, col in enumerate(y.columns):
            col_relative_errors = np.abs(
                (test_pred[:, idx] - y_test_original[:, idx]) / y_test_original[:, idx]
            )
            col_acc = np.mean(col_relative_errors <= TRAIN_PARAM_TOLERANCE)
            accuracy_dict[col] = col_acc

        relative_errors = np.abs((test_pred - y_test_original) / y_test_original)
        overall_acc = np.mean(relative_errors <= TRAIN_PARAM_TOLERANCE)

        # ▁▂▃▄ 保存模型和预处理器 ▄▃▂▁
        model_save_path = os.path.join(self._output_dir_path,
                                       ModelOutputFileName.get_model_params_file_name(model_type,
                                                                                      with_server_usage_limit_input))
        torch.save(model.state_dict(), model_save_path)

        joblib.dump(
            url_encoder,
            os.path.join(self._output_dir_path,
                         ModelOutputFileName.get_input_url_encoder_file_name(model_type,
                                                                             with_server_usage_limit_input))
        )
        if len(log_features) > 0:  # 当存在对数特征时
            joblib.dump(
                X_log_scaler,
                os.path.join(self._output_dir_path,
                             ModelOutputFileName.get_input_log_scaler_file_name(model_type,
                                                                                with_server_usage_limit_input)),
            )
        joblib.dump(
            X_other_scaler,
            os.path.join(self._output_dir_path,
                         ModelOutputFileName.get_input_other_scaler_file_name(model_type,
                                                                              with_server_usage_limit_input)),
        )
        joblib.dump(y_scaler,
                    os.path.join(self._output_dir_path,
                                 ModelOutputFileName.get_output_scaler_file_name(model_type,
                                                                                 with_server_usage_limit_input)))

        # 打印最终结果
        result_info = ["\n=== 测试结果 ==="]
        for col, acc in accuracy_dict.items():
            result_info.append(f"{col:<20}: {acc * 100:.1f}%")
        result_info.append(f"整体准确率: {overall_acc * 100:.1f}%")
        self._log('\n'.join(result_info))

        return accuracy_dict

    def _log(self, msg: str):
        if self._logger is None:
            return

        self._logger.info(msg)
