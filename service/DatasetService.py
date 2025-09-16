import os
import uuid
from io import BytesIO
from datetime import datetime

from bson import ObjectId

from model.database import get_collection
from service.ServiceValueError import ServiceValueError
from service.BusinessService import BusinessServiceInstance
import helper.csv_helper as csv_helper
from helper.model_helper import get_query_dict, get_paged_list_data, check_datetime_for_datetime, check_object_id

UPLOAD_DIR_PATH_RELATIVE_TO_CURRENT = '../dataset_files'

business_service = BusinessServiceInstance


class DatasetService:
    def __init__(self):
        self._dataset_collection = get_collection('dataset')

    async def exists_dataset_with_name(self, dataset_name: str, exclude_dataset_id: str | None = None) -> bool:
        if exclude_dataset_id is None:
            item = await self._dataset_collection.find_one({"name": dataset_name})
        else:
            item = await self._dataset_collection.find_one({"name": dataset_name,
                                                           '_id': {'$ne': ObjectId(exclude_dataset_id)}})
        return item is not None

    async def add_dataset(self,
                          name: str,
                          description: str,
                          create_time: datetime,
                          dataset_file: BytesIO,
                          dataset_file_name: str,
                          dataset_file_content_type: str) -> dict:
        if await self.exists_dataset_with_name(name):
            raise ServiceValueError('existed', 'name')

        dataset_info = csv_helper.get_csv_data_info(dataset_file)
        line_count = dataset_info['count']
        if line_count == 0:
            raise ServiceValueError('empty', 'dataset_file')
        elif line_count < 0:
            raise ServiceValueError('invalid', 'dataset_file')

        url_list = dataset_info['urls']

        file_name = self.save_dataset_file(dataset_file)
        original_file_name = dataset_file_name
        content_type = dataset_file_content_type

        item_dict = {
            'name': name,
            'description': description,
            'create_time': check_datetime_for_datetime(create_time),

            'dataset_file_name': file_name,
            'dataset_original_file_name': original_file_name,
            'dataset_content_type': content_type,
            'dataset_line_count': line_count,
            'dataset_url_list': url_list
        }

        insert_result = await self._dataset_collection.insert_one(item_dict)
        created_item = await self._dataset_collection.find_one({"_id": insert_result.inserted_id})
        return created_item

    async def get_dataset_paged_list(self,
                                     name: str | None,
                                     start_time: datetime | None,
                                     end_time: datetime | None,
                                     page: int = 1,
                                     page_size: int = 10) -> list[dict]:
        query = get_query_dict(name, start_time, end_time)

        aggregate_pipeline = [
            {'$match': query},
            {'$sort': {'create_time': -1, '_id': -1}},
            {'$lookup': {'from': "business", 'localField': "_id", 'foreignField': "dataset_id", 'as': "business_info"}},
            {'$addFields': {'business_count': {'$size': '$business_info'}}},
            {'$project': {'business_info': 0}},
        ]

        paged_list_data = await get_paged_list_data(page, page_size,
                                                    self._dataset_collection.count_documents(query),
                                                    None,
                                                    aggregate_pipeline,
                                                    self._dataset_collection.aggregate)
        return paged_list_data

    async def get_example_dataset_file_info(self) -> (str, str):
        file_name = 'example.csv'
        file_path = self.get_dataset_file_path(file_name)
        return file_name, file_path

    async def get_dataset(self, dataset_id: str) -> dict:
        item = await self._dataset_collection.find_one({"_id": ObjectId(dataset_id)})
        return item

    async def update_dataset(self,
                             dataset_id: str,
                             name: str | None,
                             description: str | None,
                             create_time: datetime | None,
                             dataset_file: BytesIO | None,
                             dataset_file_name: str | None,
                             dataset_file_content_type: str | None) -> dict:
        update_data = {}
        if name:
            if await self.exists_dataset_with_name(name, dataset_id):
                raise ServiceValueError('existed', 'name')

            update_data['name'] = name
        if description is not None:
            update_data['description'] = description
        if create_time:
            update_data['create_time'] = check_datetime_for_datetime(create_time)
        if dataset_file:
            # delete previous file
            item = await self._dataset_collection.find_one({"_id": ObjectId(dataset_id)})
            if item:
                # delete old file
                self.delete_dataset_file(item['dataset_file_name'])

                dataset_info = csv_helper.get_csv_data_info(dataset_file)
                line_count = dataset_info['count']
                if line_count == 0:
                    raise ServiceValueError('empty', 'dataset_file')
                elif line_count < 0:
                    raise ServiceValueError('invalid', 'dataset_file')

                url_list = dataset_info['urls']

                # save new file
                file_name = self.save_dataset_file(dataset_file)
                original_file_name = dataset_file_name
                content_type = dataset_file_content_type

                update_data['dataset_file_name'] = file_name
                update_data['dataset_original_file_name'] = original_file_name
                update_data['dataset_content_type'] = content_type
                update_data['dataset_line_count'] = line_count
                update_data['dataset_url_list'] = url_list

        if not update_data:
            raise ServiceValueError('invalid', 'update_data')

        await self._dataset_collection.update_one(
            {"_id": ObjectId(dataset_id)},
            {"$set": update_data}
        )
        updated_item = await self._dataset_collection.find_one({"_id": ObjectId(dataset_id)})
        return updated_item

    async def delete_dataset(self, dataset_id: str) -> bool:
        result = await self._dataset_collection.delete_one({"_id": ObjectId(dataset_id)})
        if result.deleted_count == 1:
            # delete related business
            await business_service.delete_business_list_by_dataset_id(dataset_id)
            return True
        return False

    async def get_dataset_file_info(self, dataset_id: str) -> (str, str):
        item = await self._dataset_collection.find_one({"_id": ObjectId(dataset_id)})
        if item:
            file_path = self.get_dataset_file_path(item['dataset_file_name'])
            return item['dataset_original_file_name'], file_path
        return None, None

    async def is_dataset_id_valid(self, dataset_id: str) -> bool:
        is_valid = await check_object_id(dataset_id, self._dataset_collection)
        return is_valid

    async def get_dataset_id_list_by_name(self, dataset_name: str):
        dataset_id_list = []
        async for dataset in self._dataset_collection.find({'name': {'$regex': dataset_name, '$options': 'i'}},
                                                           {'_id': 1}):
            dataset_id_list.append(dataset['_id'])
        return dataset_id_list

    def get_dataset_file_path(self, file_name: str):
        file_path = os.path.join(self._get_dataset_upload_dir_path(), file_name)
        return file_path

    def save_dataset_file(self, file: BytesIO):
        file_name = str(uuid.uuid4())
        file_path = os.path.join(self._get_dataset_upload_dir_path(), file_name)
        file.seek(0)
        with open(file_path, "wb") as buffer:
            buffer.write(file.read())
        return file_name

    def delete_dataset_file(self, file_name: str):
        file_path = os.path.join(self._get_dataset_upload_dir_path(), file_name)
        try:
            os.remove(file_path)
        except OSError:
            pass

    @staticmethod
    def _get_dataset_upload_dir_path():
        current_dir_path = os.path.dirname(os.path.abspath(__file__))
        dir_path = os.path.join(current_dir_path, UPLOAD_DIR_PATH_RELATIVE_TO_CURRENT)
        return dir_path


DatasetServiceInstance = DatasetService()
