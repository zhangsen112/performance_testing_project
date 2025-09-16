import os

from motor.motor_asyncio import AsyncIOMotorClient


MONGO_DB_URL = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
MONGO_DB_NAME = "predication_platform"
client = AsyncIOMotorClient(MONGO_DB_URL, tz_aware=True)

db = client[MONGO_DB_NAME]


def get_collection(collection_name):
    return db[collection_name]
