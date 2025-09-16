from io import BytesIO

import pandas as pd
from pandas import DataFrame

from prediction.ModelDataDefine import MODEL_DATA_SCHEMA_COLUMN_NAME_SET


def get_csv_data_info(buffer: BytesIO) -> dict:
    buffer.seek(0)
    df = pd.read_csv(buffer)

    count = -1
    urls = []

    good = check_csv_schema(df)
    if good:
        count = df.shape[0]
        urls = list(df['URL'].unique())
        urls.sort()

    return {'count': count, 'urls': urls}


def check_csv_schema(data_frame: DataFrame) -> bool:
    if MODEL_DATA_SCHEMA_COLUMN_NAME_SET.issubset(data_frame.columns):
        return True
    return False
