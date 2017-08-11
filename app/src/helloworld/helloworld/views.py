from django.http import HttpResponse, JsonResponse
import json
import requests

from helloworld import settings


def index(request):
    return HttpResponse("Hello Hasura World!")

def recieve_call(request):
    response = "<?xml version='1.0' encoding='UTF-8'?><Response><playtext>Thank you for calling Voura. We are always here to help you</playtext></Response>"
    return HttpResponse(response)

