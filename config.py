"""This module is to configure app to connect with database."""

from pymongo import MongoClient

DEBUG = True
client = MongoClient('mongodb://user1:123@localhost:27017/mongodb',authSource="admin")