from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import uuid
import shutil
import logging

from bson import ObjectId

from model.database import get_collection
from helper.model_helper import (check_datetime, get_query_dict, get_paged_list_data,
                                 check_datetime_for_datetime)
from service.ServiceValueError import ServiceValueError
from service.DatasetService import DatasetServiceInstance
from service.BusinessService import BusinessServiceInstance
from prediction.TrainManager import TrainManager

TRAIN_DIR_PATH_RELATIVE_TO_CURRENT = '../train_files'
TRAIN_LOG_FILE_NAME = 'log.txt'

dataset_service = DatasetServiceInstance
business_service = BusinessServiceInstance


class TrainService:
    def __init__(self):
        self._train_collection = get_collection('train')

        self.thread_pool_executor = ThreadPoolExecutor(max_workers=1)

    def __del__(self):
        self.thread_pool_executor.shutdown(wait=False, cancel_futures=True)

    async def exists_train_with_name(self, train_name: str, exclude_train_id: str | None = None) -> bool:
        if exclude_train_id is None:
            item = await self._train_collection.find_one({"name": train_name})
        else:
            item = await self._train_collection.find_one({"name": train_name,
                                                         '_id': {'$ne': ObjectId(exclude_train_id)}})
        return item is not None

    async def add_train(self, train_item_dict: dict) -> dict:
        if await self.exists_train_with_name(train_item_dict['name']):
            raise ServiceValueError('existed', 'name')

        if not await dataset_service.is_dataset_id_valid(train_item_dict['dataset_id']):
            raise ServiceValueError('invalid', 'dataset_id')

        item_dict = train_item_dict.copy()
        item_dict['dataset_id'] = ObjectId(train_item_dict['dataset_id'])
        item_dict['state'] = 'none'
        item_dict['accuracy'] = None
        check_datetime(item_dict, 'create_time')

        insert_result = await self._train_collection.insert_one(item_dict)
        created_item = await self._train_collection.find_one({"_id": insert_result.inserted_id})
        return created_item

    async def get_train_paged_list(self,
                                   name: str | None,
                                   data_set_name: str | None,
                                   start_time: datetime | None,
                                   end_time: datetime | None,
                                   page: int = 1,
                                   page_size: int = 10) -> list[dict]:
        query = get_query_dict(name, start_time, end_time)

        dataset_id_list = None
        if data_set_name is not None:
            dataset_id_list = await dataset_service.get_dataset_id_list_by_name(data_set_name)
        if dataset_id_list:
            query['dataset_id'] = {'$in': dataset_id_list}

        aggregate_pipeline = [
            {'$match': query},
            {'$sort': {'create_time': -1, '_id': -1}},
            {'$lookup': {'from': "dataset", 'localField': "dataset_id", 'foreignField': "_id", 'as': "dataset_info"}},

            # do not show the items whose dataset is deleted
            # {'$unwind': "$dataset_info"},
            # {'$addFields': {'dataset_name': "$dataset_info.name"}},

            # show the items whose dataset is deleted, but not with dataset_name property
            # {'$unwind': {"path": "$dataset_info", "preserveNullAndEmptyArrays": True}},
            # {'$addFields': {'dataset_name': "$dataset_info.name"}},

            # left join
            {'$addFields': {'dataset_name': {'$ifNull': [{'$arrayElemAt': ["$dataset_info.name", 0]}, None]}}},

            {'$project': {'dataset_info': 0, 'output_dir_name': 0}},
        ]

        paged_list_data = await get_paged_list_data(page, page_size,
                                                    self._train_collection.count_documents(query),
                                                    None,
                                                    aggregate_pipeline,
                                                    self._train_collection.aggregate)
        return paged_list_data

    async def get_trained_model_list(self, model_type: str) -> list[dict]:
        train_list = self._train_collection.find({'state': 'success', 'model_type': model_type}).sort({'name': 1})

        model_property_name_list = ['_id', 'name']
        model_property_name_mapped_dict = {'_id': 'train_id'}
        business_property_name_list = ['_id', 'name', 'sql']
        business_property_name_mapped_dict = {'_id': 'business_id'}

        trained_model_list = []
        async for item in train_list:
            trained_model = {(k if k not in model_property_name_mapped_dict else model_property_name_mapped_dict[k]): item[k]
                             for k in model_property_name_list}

            dataset_id = str(item['dataset_id'])
            business_list = await business_service.get_business_list(dataset_id)

            trained_model['business_list'] = [
                {(k if k not in business_property_name_mapped_dict else business_property_name_mapped_dict[k]): business[k]
                 for k in business_property_name_list}
                for business in business_list
            ]

            trained_model_list.append(trained_model)

        return trained_model_list

    async def get_train(self, train_id: str, all_fields: bool = False) -> dict | None:
        if all_fields:
            item = await self._train_collection.find_one({"_id": ObjectId(train_id)})
        else:
            item = await self._train_collection.find_one({"_id": ObjectId(train_id)}, {'output_dir_name': 0})
        return item

    async def update_train(self,
                           train_id: str,
                           name: str | None,
                           dataset_id: str | None,
                           model_type: str | None,
                           description: str | None,
                           create_time: datetime | None) -> dict:
        update_data = {}
        if name:
            if await self.exists_train_with_name(name, train_id):
                raise ServiceValueError('existed', 'name')

            update_data['name'] = name
        if dataset_id:
            if not await dataset_service.is_dataset_id_valid(dataset_id):
                raise ServiceValueError('invalid', 'dataset_id')

            update_data['dataset_id'] = ObjectId(dataset_id)
        if model_type:
            update_data['model_type'] = model_type
        if description is not None:
            update_data['description'] = description
        if create_time:
            update_data['create_time'] = check_datetime_for_datetime(create_time)

        if not update_data:
            raise ServiceValueError('invalid', 'update_data')

        await self._train_collection.update_one(
            {"_id": ObjectId(train_id)},
            {"$set": update_data}
        )
        updated_item = await self._train_collection.find_one({"_id": ObjectId(train_id)}, {'output_dir_name': 0})
        return updated_item

    async def delete_train(self, train_id: str) -> bool:
        result = await self._train_collection.delete_one({"_id": ObjectId(train_id)})
        if result.deleted_count == 1:
            train_dir_path = self._get_train_parent_dir_path(train_id)
            if os.path.isdir(train_dir_path):
                shutil.rmtree(train_dir_path)
            return True
        return False

    async def start_train(self, train_id: str) -> str | None:
        item = await self.get_train(train_id, True)
        if item is None:
            return None

        state = item['state']
        if state == 'pending' or state == 'running':
            raise ServiceValueError('invalid', 'state')
        
        if not await dataset_service.is_dataset_id_valid(str(item['dataset_id'])):
            raise ServiceValueError('invalid', 'dataset_id')

        # update state
        new_state = 'pending'
        await self.update_train_state(train_id, new_state)

        # no need to await the created task
        _ = asyncio.create_task(self._start_train_task(train_id, item))

        return new_state

    async def get_train_state(self, train_id: str) -> str | None:
        item = await self._train_collection.find_one({"_id": ObjectId(train_id)}, {'state': 1})
        if item is None:
            return None

        return item['state']

    async def update_train_state(self, train_id: str, new_state: str):
        await self._train_collection.update_one({"_id": ObjectId(train_id)}, {"$set": {'state': new_state}})

    async def check_train_state(self):
        await self._train_collection.update_many({'state': {'$in': ['pending', 'running']}},
                                                 {'$set': {
                                                     'state': 'failed',
                                                     # 'accuracy': None,
                                                 }})

    async def get_train_log(self, train_id: str) -> str:
        no_log_content_message = '暂无日志'

        item = await self.get_train(train_id, True)
        if item is None:
            return no_log_content_message
        if 'output_dir_name' not in item:
            return no_log_content_message

        output_dir_name = item['output_dir_name']
        log_file_path = TrainService._get_log_file_path(train_id, output_dir_name)

        try:
            with open(log_file_path, "r", encoding="utf-8") as file:
                content = file.read()
            return content
        except FileNotFoundError:
            return no_log_content_message

    async def get_train_dir_path(self, train_id: str) -> str | None:
        item = await self._train_collection.find_one({"_id": ObjectId(train_id), 'state': 'success'},
                                                     {'output_dir_name': 1})
        if item is None:
            return None

        output_dir_name = item['output_dir_name']
        train_dir_path = self._get_train_output_dir_path(train_id, output_dir_name)
        return train_dir_path

    async def _start_train_task(self, train_id: str, train_item_dict: dict):
        accuracy_result = None
        new_output_dir_name = None

        dataset_id = train_item_dict['dataset_id']
        dataset_file_name, dataset_file_path = await dataset_service.get_dataset_file_info(dataset_id)
        model_type = train_item_dict['model_type']
        current_output_dir_name = train_item_dict['output_dir_name'] if 'output_dir_name' in train_item_dict else None
        if dataset_file_path is not None:
            concurrent_future = self.thread_pool_executor.submit(self._train_core,
                                                                 asyncio.get_running_loop(),
                                                                 train_id,
                                                                 current_output_dir_name,
                                                                 dataset_file_path,
                                                                 model_type)
            asyncio_future = asyncio.wrap_future(concurrent_future)
            done_list, _ = await asyncio.wait([asyncio_future])
            train_task = next(iter(done_list))
            if not train_task.exception():
                accuracy_result, new_output_dir_name = train_task.result()
            # else:
            #     print(train_task.exception())

        # update train task
        to_update_state = 'success' if accuracy_result is not None else 'failed'
        to_update_output_dir_name = new_output_dir_name if new_output_dir_name is not None else current_output_dir_name
        await self._train_collection.update_one({"_id": ObjectId(train_id)},
                                                {"$set": {
                                                    'state': to_update_state,
                                                    'accuracy': accuracy_result,
                                                    'output_dir_name': to_update_output_dir_name,
                                                }})

    def _train_core(self, loop: asyncio.AbstractEventLoop,
                    train_id: str,
                    current_output_dir_path: str | None,
                    dataset_file_path: str,
                    model_type: str) -> (float | None, str | None):
        acc = None

        # update train state -> running
        loop.create_task(self.update_train_state(train_id, 'running'))

        # new output dir
        new_output_dir_name = str(uuid.uuid4())
        output_dir_path = self._get_train_output_dir_path(train_id, new_output_dir_name)
        os.makedirs(output_dir_path, exist_ok=True)

        # logger
        logger = self._get_logger(train_id, new_output_dir_name)

        try:
            # train!!
            train_manager = TrainManager(dataset_file_path, output_dir_path, logger)
            if model_type == 'app_capacity':
                acc = train_manager.train_app_capacity_models()
            elif model_type == 'server_capacity':
                acc = train_manager.train_server_capacity_models()

            # ensure the new and old output dir will remain in disk, and delete other dirs
            exclude_dirs = {new_output_dir_name}
            if current_output_dir_path is not None:
                exclude_dirs.add(current_output_dir_path)
            parent_dir_path = self._get_train_parent_dir_path(train_id)
            self._delete_dirs_except(parent_dir_path, exclude_dirs)
        except Exception as e:
            logger.error(str(e))

        return acc, new_output_dir_name

    @staticmethod
    def _delete_dirs_except(dir_path: str, exclude_dir_set: set[str]):
        if not os.path.exists(dir_path):
            return

        for item in os.listdir(dir_path):
            if item not in exclude_dir_set:
                item_path = os.path.join(dir_path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)

    @staticmethod
    def _get_train_output_dir_path(train_id: str, dir_name: str):
        parent_dir_path = TrainService._get_train_parent_dir_path(train_id)
        dir_path = os.path.join(parent_dir_path, dir_name)
        return dir_path

    @staticmethod
    def _get_train_parent_dir_path(train_id: str):
        dir_path = os.path.join(TrainService._get_train_root_dir_path(), train_id)
        return dir_path

    @staticmethod
    def _get_train_root_dir_path():
        current_dir_path = os.path.dirname(os.path.abspath(__file__))
        dir_path = os.path.join(current_dir_path, TRAIN_DIR_PATH_RELATIVE_TO_CURRENT)
        return dir_path

    @staticmethod
    def _get_log_file_path(train_id: str, train_dir_name: str):
        log_file_path = os.path.join(TrainService._get_train_output_dir_path(train_id, train_dir_name),
                                     TRAIN_LOG_FILE_NAME)
        return log_file_path

    @staticmethod
    def _get_logger(train_id: str, train_dir_name: str) -> logging.Logger:
        logger = logging.getLogger(train_dir_name)
        logger.setLevel(logging.INFO)

        # file handler
        log_file_path = TrainService._get_log_file_path(train_id, train_dir_name)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)

        # console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 将handler添加到logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


TrainServiceInstance = TrainService()
