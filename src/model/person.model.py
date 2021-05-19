
from mongoengine import  *

class User(Document):
    name = StringField()
    age = StringField()

class Person:
    name: str
    age: str