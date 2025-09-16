from enum import Enum


class ModelType(str, Enum):
    APP_CAPACITY_HTTP = 'app_capacity_http'
    APP_CAPACITY_SQL = 'app_capacity_sql'
    APP_CAPACITY_REDIS = 'app_capacity_redis'
    APP_CAPACITY_DNS = 'app_capacity_dns'

    SERVER_CAPACITY_HTTP = 'server_capacity_http'
    SERVER_CAPACITY_SQL = 'server_capacity_sql'
    SERVER_CAPACITY_REDIS = 'server_capacity_redis'
    SERVER_CAPACITY_DNS = 'server_capacity_dns'
    SERVER_CAPACITY_QPS = 'server_capacity_qps'
