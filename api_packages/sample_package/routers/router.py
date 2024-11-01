import io

from fastapi import Response, APIRouter

ROUTER = APIRouter(
    prefix="/sample",
    tags=["This is a sample PackageRouter"]
)

@ROUTER.get("/test/")
def test():
    return Response("Hello")
