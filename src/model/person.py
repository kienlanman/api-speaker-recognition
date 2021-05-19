class Person(object):
    def __init__(name, age):
        self.name = name
        self.age = age

    def to_document(self):
        return dict(
            name = self.name,
            age = self.bd,
        )

    @classmethod
    def from_document(cls, doc):
        return cls(
            name = doc['name'],
            age = doc['age'],
        )