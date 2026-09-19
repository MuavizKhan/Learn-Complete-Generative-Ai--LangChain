from typing import TypedDict

class Person(TypedDict):

    name: str
    age: int


new_person: Person = {'name': 'muaviz', 'age':23}

print(new_person)