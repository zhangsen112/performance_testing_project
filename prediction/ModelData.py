import pandas as pd

import prediction.ModelDataDefine as ModelDataDefine
from prediction.ModelType import ModelType


class ModelData:
    def __init__(self, model_type: ModelType, with_server_usage_limit_input: bool, data_df: pd.DataFrame = None):
        self.model_type = model_type
        self.with_server_usage_limit_input = with_server_usage_limit_input

        if model_type == ModelType.APP_CAPACITY_HTTP:
            self.model_data_url_input_column_names = list(ModelDataDefine.APP_CAPACITY_HTTP_MODEL_URL_INPUT_COLUMN_NAME_LIST)
            self.model_data_log_input_column_names = list(ModelDataDefine.APP_CAPACITY_HTTP_MODEL_LOG_INPUT_COLUMN_NAME_LIST)
            self.model_data_other_input_column_names = list(ModelDataDefine.APP_CAPACITY_HTTP_MODEL_OTHER_INPUT_COLUMN_NAME_LIST)
            self.model_data_output_column_names = list(ModelDataDefine.APP_CAPACITY_HTTP_MODEL_OUTPUT_COLUMN_NAME_LIST)

            self.model_data_filter_protocol_values = ['http']
        elif model_type == ModelType.APP_CAPACITY_SQL:
            self.model_data_url_input_column_names = list(ModelDataDefine.APP_CAPACITY_SQL_MODEL_URL_INPUT_COLUMN_NAME_LIST)
            self.model_data_log_input_column_names = list(ModelDataDefine.APP_CAPACITY_SQL_MODEL_LOG_INPUT_COLUMN_NAME_LIST)
            self.model_data_other_input_column_names = list(ModelDataDefine.APP_CAPACITY_SQL_MODEL_OTHER_INPUT_COLUMN_NAME_LIST)
            self.model_data_output_column_names = list(ModelDataDefine.APP_CAPACITY_SQL_MODEL_OUTPUT_COLUMN_NAME_LIST)

            self.model_data_filter_protocol_values = ['oracle', 'mysql']
        elif model_type == ModelType.APP_CAPACITY_REDIS:
            self.model_data_url_input_column_names = list(ModelDataDefine.APP_CAPACITY_REDIS_MODEL_URL_INPUT_COLUMN_NAME_LIST)
            self.model_data_log_input_column_names = list(ModelDataDefine.APP_CAPACITY_REDIS_MODEL_LOG_INPUT_COLUMN_NAME_LIST)
            self.model_data_other_input_column_names = list(ModelDataDefine.APP_CAPACITY_REDIS_MODEL_OTHER_INPUT_COLUMN_NAME_LIST)
            self.model_data_output_column_names = list(ModelDataDefine.APP_CAPACITY_REDIS_MODEL_OUTPUT_COLUMN_NAME_LIST)

            self.model_data_filter_protocol_values = ['redis']
        elif model_type == ModelType.APP_CAPACITY_DNS:
            self.model_data_url_input_column_names = list(ModelDataDefine.APP_CAPACITY_DNS_MODEL_URL_INPUT_COLUMN_NAME_LIST)
            self.model_data_log_input_column_names = list(ModelDataDefine.APP_CAPACITY_DNS_MODEL_LOG_INPUT_COLUMN_NAME_LIST)
            self.model_data_other_input_column_names = list(ModelDataDefine.APP_CAPACITY_DNS_MODEL_OTHER_INPUT_COLUMN_NAME_LIST)
            self.model_data_output_column_names = list(ModelDataDefine.APP_CAPACITY_DNS_MODEL_OUTPUT_COLUMN_NAME_LIST)

            self.model_data_filter_protocol_values = ['dns']
        elif model_type == ModelType.SERVER_CAPACITY_HTTP:
            self.model_data_url_input_column_names = list(ModelDataDefine.SERVER_CAPACITY_HTTP_MODEL_URL_INPUT_COLUMN_NAME_LIST)
            self.model_data_log_input_column_names = list(ModelDataDefine.SERVER_CAPACITY_HTTP_MODEL_LOG_INPUT_COLUMN_NAME_LIST)
            self.model_data_other_input_column_names = list(ModelDataDefine.SERVER_CAPACITY_HTTP_MODEL_OTHER_INPUT_COLUMN_NAME_LIST)
            self.model_data_output_column_names = list(ModelDataDefine.SERVER_CAPACITY_HTTP_MODEL_OUTPUT_COLUMN_NAME_LIST)

            self.model_data_filter_protocol_values = ['http']
        elif model_type == ModelType.SERVER_CAPACITY_SQL:
            self.model_data_url_input_column_names = list(ModelDataDefine.SERVER_CAPACITY_SQL_MODEL_URL_INPUT_COLUMN_NAME_LIST)
            self.model_data_log_input_column_names = list(ModelDataDefine.SERVER_CAPACITY_SQL_MODEL_LOG_INPUT_COLUMN_NAME_LIST)
            self.model_data_other_input_column_names = list(ModelDataDefine.SERVER_CAPACITY_SQL_MODEL_OTHER_INPUT_COLUMN_NAME_LIST)
            self.model_data_output_column_names = list(ModelDataDefine.SERVER_CAPACITY_SQL_MODEL_OUTPUT_COLUMN_NAME_LIST)

            self.model_data_filter_protocol_values = ['oracle', 'mysql']
        elif model_type == ModelType.SERVER_CAPACITY_REDIS:
            self.model_data_url_input_column_names = list(ModelDataDefine.SERVER_CAPACITY_REDIS_MODEL_URL_INPUT_COLUMN_NAME_LIST)
            self.model_data_log_input_column_names = list(ModelDataDefine.SERVER_CAPACITY_REDIS_MODEL_LOG_INPUT_COLUMN_NAME_LIST)
            self.model_data_other_input_column_names = list(ModelDataDefine.SERVER_CAPACITY_REDIS_MODEL_OTHER_INPUT_COLUMN_NAME_LIST)
            self.model_data_output_column_names = list(ModelDataDefine.SERVER_CAPACITY_REDIS_MODEL_OUTPUT_COLUMN_NAME_LIST)

            self.model_data_filter_protocol_values = ['redis']
        elif model_type == ModelType.SERVER_CAPACITY_DNS:
            self.model_data_url_input_column_names = list(ModelDataDefine.SERVER_CAPACITY_DNS_MODEL_URL_INPUT_COLUMN_NAME_LIST)
            self.model_data_log_input_column_names = list(ModelDataDefine.SERVER_CAPACITY_DNS_MODEL_LOG_INPUT_COLUMN_NAME_LIST)
            self.model_data_other_input_column_names = list(ModelDataDefine.SERVER_CAPACITY_DNS_MODEL_OTHER_INPUT_COLUMN_NAME_LIST)
            self.model_data_output_column_names = list(ModelDataDefine.SERVER_CAPACITY_DNS_MODEL_OUTPUT_COLUMN_NAME_LIST)

            self.model_data_filter_protocol_values = ['dns']
        elif model_type == ModelType.SERVER_CAPACITY_QPS:
            self.model_data_url_input_column_names = list(ModelDataDefine.SERVER_CAPACITY_QPS_MODEL_URL_INPUT_COLUMN_NAME_LIST)
            self.model_data_log_input_column_names = list(ModelDataDefine.SERVER_CAPACITY_QPS_MODEL_LOG_INPUT_COLUMN_NAME_LIST)
            self.model_data_other_input_column_names = list(ModelDataDefine.SERVER_CAPACITY_QPS_MODEL_OTHER_INPUT_COLUMN_NAME_LIST)
            self.model_data_output_column_names = list(ModelDataDefine.SERVER_CAPACITY_QPS_MODEL_OUTPUT_COLUMN_NAME_LIST)

            self.model_data_filter_protocol_values = ['http']

        if with_server_usage_limit_input:
            self.model_data_other_input_column_names += ModelDataDefine.SERVER_USAGE_COLUMN_NAME_LIST
        else:
            self.model_data_output_column_names += ModelDataDefine.SERVER_USAGE_COLUMN_NAME_LIST

        self.model_data_input_column_names = self.model_data_url_input_column_names + self.model_data_log_input_column_names + self.model_data_other_input_column_names

        if data_df is not None:
            filtered_data_df = data_df[data_df[ModelDataDefine.MODEL_DATA_SCHEMA_FILTER_PROTOCOL_COLUMN_NAME].isin(
                self.model_data_filter_protocol_values)]

            self.model_data_url_input_df = filtered_data_df[self.model_data_url_input_column_names]
            self.model_data_log_input_df = filtered_data_df[self.model_data_log_input_column_names]
            self.model_data_other_input_df = filtered_data_df[self.model_data_other_input_column_names]
            self.model_data_input_df = pd.concat([self.model_data_url_input_df,
                                                  self.model_data_log_input_df,
                                                  self.model_data_other_input_df], axis=1)
            self.model_data_output_df = filtered_data_df[self.model_data_output_column_names]
