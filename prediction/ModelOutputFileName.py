from prediction.ModelType import ModelType

MODEL_PARAMS_FILE_NAME = 'model.pth'
INPUT_URL_ENCODER_FILE_NAME = 'url_encoder.pkl'
INPUT_LOG_SCALER_FILE_NAME = 'X_log_scaler.pkl'
INPUT_OTHER_SCALER_FILE_NAME = 'X_other_scaler.pkl'
OUTPUT_SCALER_FILE_NAME = 'y_scaler.pkl'


def get_model_name(model_type: ModelType, with_server_usage_limit_input: bool) -> str:
    return f'{model_type.value}{"_limit" if with_server_usage_limit_input else ""}'


def get_model_params_file_name(model_type: ModelType, with_server_usage_limit_input: bool) -> str:
    return f'{get_model_name(model_type, with_server_usage_limit_input)}_{MODEL_PARAMS_FILE_NAME}'


def get_input_url_encoder_file_name(model_type: ModelType, with_server_usage_limit_input: bool) -> str:
    return f'{get_model_name(model_type, with_server_usage_limit_input)}_{INPUT_URL_ENCODER_FILE_NAME}'


def get_input_log_scaler_file_name(model_type: ModelType, with_server_usage_limit_input: bool) -> str:
    return f'{get_model_name(model_type, with_server_usage_limit_input)}_{INPUT_LOG_SCALER_FILE_NAME}'


def get_input_other_scaler_file_name(model_type: ModelType, with_server_usage_limit_input: bool) -> str:
    return f'{get_model_name(model_type, with_server_usage_limit_input)}_{INPUT_OTHER_SCALER_FILE_NAME}'


def get_output_scaler_file_name(model_type: ModelType, with_server_usage_limit_input: bool) -> str:
    return f'{get_model_name(model_type, with_server_usage_limit_input)}_{OUTPUT_SCALER_FILE_NAME}'
