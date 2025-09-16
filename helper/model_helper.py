from datetime import datetime, timezone

from bson import ObjectId
from fastapi.encoders import jsonable_encoder


def get_valid_response_item(item: dict | None, json_encode: bool = True) -> dict | None:
    if item is None:
        return item

    if not isinstance(item, dict):
        return item

    good_item = item.copy()

    # _id -> id
    if "_id" in good_item:
        good_item["id"] = str(good_item["_id"])
        del good_item["_id"]

    # ObjectId -> str
    for k, v in good_item.items():
        if isinstance(v, ObjectId):
            good_item[k] = str(v)
        elif isinstance(v, list):
            good_item[k] = get_valid_response_list(v, json_encode)
        elif isinstance(v, dict):
            good_item[k] = get_valid_response_item(v, json_encode)

    #  json encode
    if json_encode:
        good_item = jsonable_encoder(good_item)

    return good_item


def get_valid_response_list(data: list[dict], json_encode: bool = True) -> list[dict]:
    return [get_valid_response_item(item, json_encode) for item in data]


def check_datetime(item: dict, datetime_field_name: str):
    if item is None or datetime_field_name not in item:
        return

    dt = item[datetime_field_name]

    item[datetime_field_name] = check_datetime_for_datetime(dt)


def check_datetime_for_datetime(date_time: datetime):
    if date_time is None:
        return datetime.now(timezone.utc)
    if date_time.tzinfo is None:
        return date_time.astimezone(timezone.utc)
    return date_time


async def check_object_id(object_id_str: str, collection, object_id_field_name: str = '_id'):
    if not ObjectId.is_valid(object_id_str):
        return False

    item = await collection.find_one({object_id_field_name: ObjectId(object_id_str)}, {object_id_field_name: 1})
    return item is not None


def get_query_dict(name: str | None = None, start_time: datetime | None = None, end_time: datetime | None = None):
    query = {}

    # query for name
    if name:
        # 'i' means ignore case
        query['name'] = {'$regex': name, '$options': 'i'}

    # query for time
    time_query = {}
    if start_time:
        if start_time.tzinfo is None:
            start_time_utc = start_time.astimezone(timezone.utc)
        else:
            start_time_utc = start_time
        time_query['$gte'] = start_time_utc
    if end_time:
        if end_time.tzinfo is None:
            end_time_utc = end_time.astimezone(timezone.utc)
        else:
            end_time_utc = end_time
        time_query['$lte'] = end_time_utc
    if time_query:
        query['create_time'] = time_query

    return query


async def get_paged_list_data(page: int, page_size: int,
                              total_count_async,
                              all_item_cursor_async,
                              all_item_aggregate_pipline: list = None,
                              all_item_aggregate_get_cursor_func=None):
    # skip item count
    skip = (page - 1) * page_size

    # total count
    total = await total_count_async

    all_item_cursor = None
    if all_item_cursor_async is not None:
        # find case: find -> sort -> skip -> limit
        all_item_cursor = all_item_cursor_async.skip(skip).limit(page_size)
    elif all_item_aggregate_pipline is not None and all_item_aggregate_get_cursor_func is not None:
        # aggregate case: aggregate
        sort_item_index = None
        for i in range(len(all_item_aggregate_pipline)):
            pipeline_item = all_item_aggregate_pipline[i]
            if isinstance(pipeline_item, dict) and '$sort' in pipeline_item:
                sort_item_index = i
                break
        if sort_item_index is not None:
            all_item_aggregate_pipline.insert(sort_item_index + 1, {'$skip': skip})
            all_item_aggregate_pipline.insert(sort_item_index + 2, {'$limit': page_size})
            all_item_cursor = all_item_aggregate_get_cursor_func(all_item_aggregate_pipline)

    task_list = []
    async for task in all_item_cursor:
        task_list.append(get_valid_response_item(task))

    result_data = {
        'items': task_list,
        'page': page,
        'page_size': page_size,
        'total_page_count': (total + page_size - 1) // page_size,
        'total_item_count': total,
    }
    return result_data
