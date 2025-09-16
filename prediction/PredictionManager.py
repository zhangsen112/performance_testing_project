import math

import pandas as pd

from prediction.ServerConfig import ServerConfig, MIN_CPU_CORES, MIN_MEM_SIZES, MIN_DISK_SIZES
from prediction.ServerUsage import ServerUsage
from prediction.AppCapacityResult import AppCapacityResult
from prediction.ServerCapacityResult import ServerCapacityResult
from prediction.PerformanceIndexInfo import PerformanceIndexInfo
from prediction.MLPModelManager import MLPModelManager
from prediction.ModelType import ModelType
from prediction.ModelData import ModelData

PredictionTaskTypeAppCapacity = 1
PredictionTaskTypeServerCapacity = 2


class PredictionManager:
    def __init__(self):
        pass

    def predict(self, task_type: int, model_dir_path: str, input_params: dict) -> dict:
        print(f'---------- prediction task ({task_type}) params ----------')
        print(input_params)

        if task_type == PredictionTaskTypeAppCapacity:
            server_config = []
            for item in input_params['server_config']:
                limit = ServerConfig()
                limit.cpu = item['cpu']
                limit.memory = item['memory']
                limit.disk = item['disk']
                server_config.append(limit)

            server_usage_limit = self._get_server_usage_limit_from_data(input_params['server_usage_limit'])

            business_index = input_params['business_index']
            if business_index in ['http_response_time', 'qps', 'success_rate', 'business_traffic']:
                result = self.predict_app_capacity_http(model_dir_path,
                                                        input_params['url'],
                                                        input_params['requests_per_second'],
                                                        server_config,
                                                        server_usage_limit)
                return self._get_dict_from_app_capacity_result(result, input_params['requests_per_second'])
            elif business_index in ['sql_response_time']:
                result = self.predict_app_capacity_sql(model_dir_path,
                                                       input_params['url'],
                                                       input_params['requests_per_second'],
                                                       server_config,
                                                       server_usage_limit)
                return self._get_dict_from_app_capacity_result(result, input_params['requests_per_second'])
            elif business_index in ['redis_response_time']:
                result = self.predict_app_capacity_redis(model_dir_path,
                                                         input_params['url'],
                                                         input_params['requests_per_second'],
                                                         server_config,
                                                         server_usage_limit)
                return self._get_dict_from_app_capacity_result(result, input_params['requests_per_second'])
            elif business_index in ['dns_response_time']:
                result = self.predict_app_capacity_dns(model_dir_path,
                                                       input_params['url'],
                                                       input_params['requests_per_second'],
                                                       server_config,
                                                       server_usage_limit)
                return self._get_dict_from_app_capacity_result(result, input_params['requests_per_second'])
        elif task_type == PredictionTaskTypeServerCapacity:
            server_usage_limit = self._get_server_usage_limit_from_data(input_params['server_usage_limit'])

            business_index = input_params['business_index']
            if business_index in ['http_response_time']:
                result = self.predict_server_capacity_http(model_dir_path,
                                                           input_params['url'],
                                                           input_params['requests_per_second'],
                                                           input_params['success_rate'],
                                                           input_params['http_response_time'],
                                                           server_usage_limit)
                return self._get_dict_from_server_capacity_result(result)
            elif business_index in ['sql_response_time']:
                result = self.predict_server_capacity_sql(model_dir_path,
                                                          input_params['url'],
                                                          input_params['requests_per_second'],
                                                          input_params['success_rate'],
                                                          input_params['sql_response_time'],
                                                          server_usage_limit)
                return self._get_dict_from_server_capacity_result(result)
            elif business_index in ['redis_response_time']:
                result = self.predict_server_capacity_redis(model_dir_path,
                                                            input_params['url'],
                                                            input_params['requests_per_second'],
                                                            input_params['success_rate'],
                                                            input_params['redis_response_time'],
                                                            server_usage_limit)
                return self._get_dict_from_server_capacity_result(result)
            elif business_index in ['dns_response_time']:
                result = self.predict_server_capacity_dns(model_dir_path,
                                                          input_params['url'],
                                                          input_params['requests_per_second'],
                                                          input_params['success_rate'],
                                                          input_params['dns_response_time'],
                                                          server_usage_limit)
                return self._get_dict_from_server_capacity_result(result)
            elif business_index in ['qps']:
                result = self.predict_server_capacity_qps(model_dir_path,
                                                          input_params['url'],
                                                          input_params['requests_per_second'],
                                                          input_params['qps'],
                                                          server_usage_limit)
                return self._get_dict_from_server_capacity_result(result)

        return {}

    def predict_app_capacity_http(self,
                                  model_dir_path: str,
                                  url: str,
                                  requests_per_second: float,
                                  server_config: list[ServerConfig],
                                  server_usage_limit: list[ServerUsage] = None) -> AppCapacityResult:
        prediction = self._predict_app_capacity_core(model_dir_path, ModelType.APP_CAPACITY_HTTP,
                                                     url, requests_per_second, server_config, server_usage_limit)

        result = AppCapacityResult()
        result.http_response_time = prediction[0][0].item()
        result.qps = prediction[0][1].item()
        result.page_load_time = prediction[0][2].item()
        result.success_rate = result.qps / requests_per_second

        if server_usage_limit is None:
            result.server_usage = self._get_app_capacity_server_usage_from_prediction(prediction, 3)

        return result

    def predict_app_capacity_sql(self,
                                 model_dir_path: str,
                                 url: str,
                                 requests_per_second: float,
                                 server_config: list[ServerConfig],
                                 server_usage_limit: list[ServerUsage] = None) -> AppCapacityResult:
        prediction = self._predict_app_capacity_core(model_dir_path, ModelType.APP_CAPACITY_SQL,
                                                     url, requests_per_second, server_config, server_usage_limit)

        result = AppCapacityResult()
        result.sql_response_time = prediction[0][0].item()

        if server_usage_limit is None:
            result.server_usage = self._get_app_capacity_server_usage_from_prediction(prediction, 1)

        return result

    def predict_app_capacity_redis(self,
                                   model_dir_path: str,
                                   url: str,
                                   requests_per_second: float,
                                   server_config: list[ServerConfig],
                                   server_usage_limit: list[ServerUsage] = None) -> AppCapacityResult:
        prediction = self._predict_app_capacity_core(model_dir_path, ModelType.APP_CAPACITY_REDIS,
                                                     url, requests_per_second, server_config, server_usage_limit)

        result = AppCapacityResult()
        result.redis_response_time = prediction[0][0].item()

        if server_usage_limit is None:
            result.server_usage = self._get_app_capacity_server_usage_from_prediction(prediction, 1)

        return result

    def predict_app_capacity_dns(self,
                                 model_dir_path: str,
                                 url: str,
                                 requests_per_second: float,
                                 server_config: list[ServerConfig],
                                 server_usage_limit: list[ServerUsage] = None) -> AppCapacityResult:
        prediction = self._predict_app_capacity_core(model_dir_path, ModelType.APP_CAPACITY_DNS,
                                                     url, requests_per_second, server_config, server_usage_limit)

        result = AppCapacityResult()
        result.dns_response_time = prediction[0][0].item()

        if server_usage_limit is None:
            result.server_usage = self._get_app_capacity_server_usage_from_prediction(prediction, 1)

        return result

    def predict_server_capacity_http(self,
                                     model_dir_path: str,
                                     url: str,
                                     requests_per_second: float,
                                     success_rate: float,
                                     response_time: float,
                                     server_usage_limit: list[ServerUsage] = None) -> ServerCapacityResult:
        params = PerformanceIndexInfo()
        params.http_response_time = response_time
        params.success_rate = success_rate
        prediction = self._predict_server_capacity_core(model_dir_path, ModelType.SERVER_CAPACITY_HTTP,
                                                        url, requests_per_second, params, server_usage_limit)

        result = self._get_server_capacity_result_from_prediction(prediction, server_usage_limit)
        return result

    def predict_server_capacity_sql(self,
                                    model_dir_path: str,
                                    url: str,
                                    requests_per_second: float,
                                    success_rate: float,
                                    response_time: float,
                                    server_usage_limit: list[ServerUsage] = None) -> ServerCapacityResult:
        params = PerformanceIndexInfo()
        params.sql_response_time = response_time
        params.success_rate = success_rate
        prediction = self._predict_server_capacity_core(model_dir_path, ModelType.SERVER_CAPACITY_SQL,
                                                        url, requests_per_second, params, server_usage_limit)

        result = self._get_server_capacity_result_from_prediction(prediction, server_usage_limit)
        return result

    def predict_server_capacity_redis(self,
                                      model_dir_path: str,
                                      url: str,
                                      requests_per_second: float,
                                      success_rate: float,
                                      response_time: float,
                                      server_usage_limit: list[ServerUsage] = None) -> ServerCapacityResult:
        params = PerformanceIndexInfo()
        params.redis_response_time = response_time
        params.success_rate = success_rate
        prediction = self._predict_server_capacity_core(model_dir_path, ModelType.SERVER_CAPACITY_REDIS,
                                                        url, requests_per_second, params, server_usage_limit)

        result = self._get_server_capacity_result_from_prediction(prediction, server_usage_limit)
        return result

    def predict_server_capacity_dns(self,
                                    model_dir_path: str,
                                    url: str,
                                    requests_per_second: float,
                                    success_rate: float,
                                    response_time: float,
                                    server_usage_limit: list[ServerUsage] = None) -> ServerCapacityResult:
        params = PerformanceIndexInfo()
        params.dns_response_time = response_time
        params.success_rate = success_rate
        prediction = self._predict_server_capacity_core(model_dir_path, ModelType.SERVER_CAPACITY_DNS,
                                                        url, requests_per_second, params, server_usage_limit)

        result = self._get_server_capacity_result_from_prediction(prediction, server_usage_limit)
        return result

    def predict_server_capacity_qps(self,
                                    model_dir_path: str,
                                    url: str,
                                    requests_per_second: float,
                                    qps: float,
                                    server_usage_limit: list[ServerUsage] = None) -> ServerCapacityResult:
        params = PerformanceIndexInfo()
        params.qps = qps
        prediction = self._predict_server_capacity_core(model_dir_path, ModelType.SERVER_CAPACITY_QPS,
                                                        url, requests_per_second, params, server_usage_limit)

        result = self._get_server_capacity_result_from_prediction(prediction, server_usage_limit)
        return result

    @staticmethod
    def _predict_app_capacity_core(model_dir_path: str,
                                   model_type: ModelType,
                                   url: str,
                                   requests_per_second: float,
                                   server_config: list[ServerConfig],
                                   server_usage_limit: list[ServerUsage] = None):
        with_server_usage_limit_input = server_usage_limit is not None
        model_manager = MLPModelManager(model_dir_path, model_type, with_server_usage_limit_input)
        model_data = ModelData(model_type, with_server_usage_limit_input)

        # url
        input_value_list = [url]
        # log
        for config in server_config:
            input_value_list.append(math.log(config.cpu))
            input_value_list.append(math.log(config.memory))
            input_value_list.append(math.log(config.disk))
        # other
        input_value_list.append(requests_per_second)
        if server_usage_limit is not None:
            for usage in server_usage_limit:
                input_value_list.append(usage.cpu_usage)
                input_value_list.append(usage.memory_usage)
                input_value_list.append(usage.disk_usage)

        input_df = pd.DataFrame([input_value_list], columns=model_data.model_data_input_column_names)
        input_url_df = input_df[model_data.model_data_url_input_column_names]
        input_log_df = input_df[model_data.model_data_log_input_column_names] if model_data.model_data_log_input_column_names else None
        input_other_df = input_df[model_data.model_data_other_input_column_names]
        prediction = model_manager.predict(input_url_df, input_log_df, input_other_df)

        return prediction

    @staticmethod
    def _predict_server_capacity_core(model_dir_path: str,
                                      model_type: ModelType,
                                      url: str,
                                      requests_per_second: float,
                                      params: PerformanceIndexInfo,
                                      server_usage_limit: list[ServerUsage] = None):
        with_server_usage_limit_input = server_usage_limit is not None
        model_manager = MLPModelManager(model_dir_path, model_type, with_server_usage_limit_input)
        model_data = ModelData(model_type, with_server_usage_limit_input)

        # url and other(requests_per_second)
        input_value_list = [url, requests_per_second]
        # other
        if params.http_response_time is not None:
            input_value_list.append(params.http_response_time)
        elif params.sql_response_time is not None:
            input_value_list.append(params.sql_response_time)
        elif params.redis_response_time is not None:
            input_value_list.append(params.redis_response_time)
        elif params.dns_response_time is not None:
            input_value_list.append(params.dns_response_time)
        elif params.qps is not None:
            input_value_list.append(params.qps)
        if params.success_rate is not None:
            input_value_list.append(params.success_rate)

        if server_usage_limit is not None:
            for usage in server_usage_limit:
                input_value_list.append(usage.cpu_usage)
                input_value_list.append(usage.memory_usage)
                input_value_list.append(usage.disk_usage)

        input_df = pd.DataFrame([input_value_list], columns=model_data.model_data_input_column_names)
        input_url_df = input_df[model_data.model_data_url_input_column_names]
        input_log_df = input_df[model_data.model_data_log_input_column_names] if model_data.model_data_log_input_column_names else None
        input_other_df = input_df[model_data.model_data_other_input_column_names]
        prediction = model_manager.predict(input_url_df, input_log_df, input_other_df)

        return prediction

    @staticmethod
    def _get_app_capacity_server_usage_from_prediction(prediction, start_index):
        server_usage = []
        for i in range(start_index, 3 * 3 + start_index, 3):
            usage = ServerUsage()
            usage.cpu_usage = prediction[0][i].item()
            usage.memory_usage = prediction[0][i + 1].item()
            usage.disk_usage = prediction[0][i + 2].item()
            server_usage.append(usage)
        return server_usage

    @staticmethod
    def _get_server_capacity_result_from_prediction(prediction, server_usage_limit: list[ServerUsage] = None) -> ServerCapacityResult:
        result = ServerCapacityResult()
        result.server_config = []
        if server_usage_limit is None:
            result.server_usage = []
        else:
            result.server_usage = None

        for i in range(3):
            prediction_cpu_value = prediction[0][3 * i].item()
            cpu_config, cpu_usage = PredictionManager._get_server_capacity_server_config_and_usage_info(
                prediction_cpu_value, MIN_CPU_CORES, PredictionManager._cpu_multiple_function,
                server_usage_limit[i].cpu_usage if server_usage_limit is not None else None)

            prediction_mem_value = prediction[0][3 * i + 1].item()
            mem_config, mem_usage = PredictionManager._get_server_capacity_server_config_and_usage_info(
                prediction_mem_value, MIN_MEM_SIZES, PredictionManager._memory_multiple_function,
                server_usage_limit[i].memory_usage if server_usage_limit is not None else None)

            prediction_disk_value = prediction[0][3 * i + 2].item()
            disk_config, disk_usage = PredictionManager._get_server_capacity_server_config_and_usage_info(
                prediction_disk_value, MIN_DISK_SIZES, PredictionManager._disk_multiple_function,
                server_usage_limit[i].disk_usage if server_usage_limit is not None else None)

            prediction_server_config = ServerConfig()
            prediction_server_config.cpu = cpu_config
            prediction_server_config.memory = mem_config
            prediction_server_config.disk = disk_config
            result.server_config.append(prediction_server_config)

            if result.server_usage is not None:
                prediction_server_usage = ServerUsage()
                prediction_server_usage.cpu_usage = cpu_usage
                prediction_server_usage.memory_usage = mem_usage
                prediction_server_usage.disk_usage = disk_usage
                result.server_usage.append(prediction_server_usage)

        return result

    @staticmethod
    def _cpu_multiple_function(cpu_core_used_value: float) -> int:
        if cpu_core_used_value < 100:
            return 2
        else:
            return 8

    @staticmethod
    def _memory_multiple_function(memory_size_used_value: float) -> int:
        if memory_size_used_value < 100:
            return 4
        else:
            return 16

    @staticmethod
    def _disk_multiple_function(disk_size_used_value: float) -> int:
        if disk_size_used_value < 100:
            return 4
        else:
            return 32

    @staticmethod
    def _get_server_capacity_server_config_and_usage_info(prediction_config_value, min_config_value, multiple_function,
                                                          usage_limit: float | None) -> (int, float | None):
        if prediction_config_value < 0:
            prediction_config_value = -prediction_config_value
        elif prediction_config_value == 0:
            prediction_config_value = min_config_value

        multiple = multiple_function(prediction_config_value)
        config_value = int(math.ceil(prediction_config_value / multiple) * multiple)
        if usage_limit is None:
            usage_value = prediction_config_value / config_value * 100
        else:
            usage_value = prediction_config_value * usage_limit / config_value * 100

        return config_value, usage_value

    @staticmethod
    def _get_dict_from_app_capacity_result(result: AppCapacityResult, requests_per_second: float) -> dict:
        result.fix_values(requests_per_second)

        data = {
            'http_response_time': result.http_response_time,
            'qps': result.qps,
            # web接口约定, success_rate是[0-100]；而model接口约定，success_rate是[0-1]
            'success_rate': result.success_rate * 100 if result.success_rate is not None else None,
            'page_load_time': result.page_load_time,
            'sql_response_time': result.sql_response_time,
            'redis_response_time': result.redis_response_time,
            'dns_response_time': result.dns_response_time,
            'server_usage': None
        }
        if result.server_usage is not None:
            data['server_usage'] = [{'cpu_usage': usage.cpu_usage,
                                     'memory_usage': usage.memory_usage,
                                     'disk_usage': usage.disk_usage} for usage in result.server_usage]

        return data

    @staticmethod
    def _get_dict_from_server_capacity_result(result: ServerCapacityResult) -> dict:
        result.fix_values()

        data = dict()
        data['server_config'] = [{'cpu': config.cpu,
                                  'memory': config.memory,
                                  'disk': config.disk} for config in result.server_config]
        if result.server_usage is not None:
            data['server_usage'] = [{'cpu_usage': usage.cpu_usage,
                                     'memory_usage': usage.memory_usage,
                                     'disk_usage': usage.disk_usage} for usage in result.server_usage]
        return data

    @staticmethod
    def _get_server_usage_limit_from_data(server_usage_limit_data):
        if server_usage_limit_data is None:
            return None

        server_usage_limit = []
        for limit in server_usage_limit_data:
            if limit is not None:
                server_usage_limit.append(
                    ServerUsage(limit['cpu_usage'] if 'cpu_usage' in limit and limit['cpu_usage'] is not None else 100,
                                limit['memory_usage'] if 'memory_usage' in limit and limit['memory_usage'] is not None else 100,
                                limit['disk_usage'] if 'disk_usage' in limit and limit['disk_usage'] is not None else 100))
            else:
                server_usage_limit.append(ServerUsage(100, 100, 100))
        return server_usage_limit


PredictionManagerInstance = PredictionManager()
