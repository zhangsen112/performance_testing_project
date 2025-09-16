from bson import ObjectId
from model.database import get_collection
from helper.model_helper import check_object_id
from service.ServiceValueError import ServiceValueError


class BusinessService:
    def __init__(self):
        self._business_collection = get_collection('business')

    async def exists_business_with_name(self, dataset_id: str,
                                        business_name: str,
                                        exclude_business_id: str | None = None) -> bool:
        if exclude_business_id is None:
            item = await self._business_collection.find_one({"name": business_name, 'dataset_id': ObjectId(dataset_id)})
        else:
            item = await self._business_collection.find_one({"name": business_name,
                                                             'dataset_id': ObjectId(dataset_id),
                                                             '_id': {'$ne': ObjectId(exclude_business_id)}})
        return item is not None

    async def is_business_id_valid(self, business_id: str) -> bool:
        is_valid = await check_object_id(business_id, self._business_collection)
        return is_valid

    async def add_business(self, business_dict: dict) -> dict:
        if await self.exists_business_with_name(business_dict['dataset_id'], business_dict['name']):
            raise ServiceValueError('existed', 'name')

        business_dict['dataset_id'] = ObjectId(business_dict['dataset_id'])

        insert_result = await self._business_collection.insert_one(business_dict)
        created_business = await self._business_collection.find_one({"_id": insert_result.inserted_id})
        return created_business

    async def get_business_list(self, dataset_id: str) -> list[dict]:
        business_list = []
        async for business in self._business_collection.find({"dataset_id": ObjectId(dataset_id)}).sort({'_id': 1}):
            business_list.append(business)
        return business_list

    async def get_business(self, business_id: str) -> dict:
        business = await self._business_collection.find_one({"_id": ObjectId(business_id)})
        return business

    async def update_business(self, business_id: str, business_dict: dict) -> dict:
        if await self.exists_business_with_name(business_dict['dataset_id'], business_dict['name'], business_id):
            raise ServiceValueError('existed', 'name')

        business_dict['dataset_id'] = ObjectId(business_dict['dataset_id'])

        await self._business_collection.update_one(
            {"_id": ObjectId(business_id)},
            {"$set": business_dict}
        )
        updated_business = await self._business_collection.find_one({"_id": ObjectId(business_id)})
        return updated_business

    async def delete_business(self, business_id: str) -> bool:
        result = await self._business_collection.delete_one({"_id": ObjectId(business_id)})
        return result.deleted_count == 1

    async def delete_business_list_by_dataset_id(self, dataset_id: str) -> bool:
        await self._business_collection.delete_many({"dataset_id": ObjectId(dataset_id)})
        return True


BusinessServiceInstance = BusinessService()
