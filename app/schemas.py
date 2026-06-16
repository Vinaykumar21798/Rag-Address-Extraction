from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip: str

class AddressList(BaseModel):
    addresses: list[Address]

class SearchRequest(BaseModel):
    question: str
    k: int = 4
    rewrite: bool = False

