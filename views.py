"""
nlab views
"""
from rest_framework import status
from .models import Layer, Network
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .nlab_utils.view_helpers import _is_subset
from django.contrib.auth.decorators import login_required

@login_required
@api_view(['POST'])
def create_layer(request, *args, **kwargs):
    """
    Creates a new nerual layer that can be used to create networks

    Inputs    
        :request: <Http.Request> contains the information needed to create a layer

    Outputs
        :returns: Status ... 
                        ... HTTP_202_ACCEPTED if the user is verified
                        ... HTTP_403_FORBIDDEN if the user is not verified
                        ... HTTP_412_PRECONDITION_FAILED if one one more of the request fields don't meet their precondition(s)
    """
    layer_fields = ["name", "store", "parameters", "pytorch", "pennylane", "tensorflow"]
    layer_status = _is_subset(layer_fields, request.data.keys())

    if layer_status == status.HTTP_200_OK:
        name       = request.data['name']
        store      = request.data['store']
        pytorch    = request.data['pytorch']
        pennylane  = request.data['pennylane']
        tensorflow = request.data['tensorflow']
        parameters = request.data['parameters']

        supports = ""
        if pytorch: supports += " pytorch"
        if pennylane: supports += " pennylane"
        if tensorflow: supports += " tensorflow"

        print(f"Creating layer {name} supporting{supports}")
        layer = Layer.objects.create(name=name, store=store, pytorch=pytorch, pennylane=pennylane, tensorflow=tensorflow, parameters=parameters)

    return Response(status = layer_status)