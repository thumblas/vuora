from django.http import HttpResponse, JsonResponse
import json
import requests

from helloworld import settings


def index(request):
    return HttpResponse("Hello Hasura World!")

def recieve_call(request):
    return HttpResponse("Hello Recieve Call")

