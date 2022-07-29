"""
nlab models
"""
import os
import json
import uuid
import torch
import torch.nn as nn
from pathlib import Path
from django.db import models

### Global Constants ###
alphabet_size = 26
pytorch = "pytorch"
pennylane = "pennylane"
tensorflow = "tensorflow"
imports = {
    pytorch: "import torch\nimport torch.nn as nn"
}
activations = {
    pytorch: {'relu': "nn.ReLu()"}
}

class Layer(models.Model):
    """
    AF(id, name, pytorch, tensorflow, pennylane) = a neural network layer along with different 
                                                    implementation formats 

    Representation Invariant
        - inherits from models.Model
    
    Representation Exposure
        - inherits from models.Model
        - access allowed to all fields but they are all immutable
    """
    ##### Representation #####
    store = models.CharField(max_length = alphabet_size**2)
    parameters = models.CharField(max_length = alphabet_size**2)
    name = models.CharField(max_length = alphabet_size, unique=True)
    pytorch = models.CharField(max_length = alphabet_size**3, blank=True, null=True)
    pennylane = models.CharField(max_length = alphabet_size**3, blank=True, null=True)
    tensorflow = models.CharField(max_length = alphabet_size**3, blank=True, null=True)
    id = models.UUIDField(primary_key = True,  editable = False, unique = True, default = uuid.uuid4)

    def construct(self, parameters, mode = pytorch):
        """
        Constructs the network and saves the script associated with executing it
        
        Inputs
            :parameters: <dict> of values to be used to initiate the layer

        Throws
            <RunTimeError> if execution takes longer than timeout
        """
        possible_parameters = [parameter[0] for parameter in json.loads(self.parameters)[mode]]

        if mode == pytorch: layer = f"{self.pytorch}("
        elif mode == tensorflow: layer = f"{self.pennylane}("
        elif mode == tensorflow: layer = f"{self.tensorflow}("
        else: raise ValueError(f"Invalid construction mode {mode}")

        for parameter, value in parameters.items():
            if parameter in possible_parameters:
                layer += f"{parameter}={value},"
        
        return layer + ")"

    def test(self, input_, weights, timeout: int = 60):
        """
        Runs the layer on an input tensor
    
        Inputs
            :input_: <torch.Tensor> of values to be inputted into the layer
            :weights: <torch.Tensor> determining the layer
            :timeout: <int> how long to run the code for before force stopping in seconds
    
        Throws
            <RunTimeError> if execution takes longer than 10s
        """
        raise NotImplementedError
    
    def store(self):
        """
        Returns code that stores the output of the layer based on self.store and prints the path to where the output was stored

        Outputs
            :returns: <str> of code indicating how layer output should be stored
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """ Override models.Model.__str__() """
        supports = ""
        if self.pytorch: supports += f" <pytorch> {len(json.loads(self.parameters)[pytorch])}"
        if self.pennylane: supports += f" <pennylane> {len(json.loads(self.parameters)[pennylane])}"
        if self.tensorflow: supports += f" <tensorflow> {len(json.loads(self.parameters)[tensorflow])}"

        return f"{self.name}:{supports}"

class Network(models.Model):
    """
    AF(id, name, layers, loss, owner, graph, type, weights) = a neural network along with its 
                                                                implementation and weights

    Representation Invariant
        - inherits from models.Model
    
    Representation Exposure
        - inherits from models.Model
        - access allowed to all fields but they are all immutable
    """
    ##### Representation #####
    name = models.CharField(max_length = alphabet_size)
    type = models.CharField(max_length = alphabet_size)
    id = models.UUIDField(primary_key = True,  editable = False, unique = True, default = uuid.uuid4)

    
    graph = models.CharField(max_length = alphabet_size**4)
    layers = models.CharField(max_length = alphabet_size**4)
    loss = models.CharField(max_length = alphabet_size, blank=True, null=True)
    weights = models.CharField(max_length = alphabet_size**2, blank=True, null=True)

    network_path = f"{Path(__file__).parent.absolute()}{os.sep}networks{os.sep}{id}.py"
    owner = models.ForeignKey('user.CustomUser', on_delete=models.CASCADE, blank=True)

    def construct(self):
        """
        Constructs the network and saves the script associated with executing it
    
        Throws
            <RunTimeError> if execution takes longer than timeout
        """
        network = f"{imports[self.type.lower()]}\n\nclass {self.name}(nn.Module):"
        layers = "\n\tdef __init__(self):\n\n\tsuper().__init__()"  
        for i, layer_info in enumerate(self.layers): 
            layer_type, parameters = layer_info
            layer = Layer.objects.get(type=layer_type)
            if not layer: raise ValueError(f"Something went wrong! Layer {i}[{layer_type}] does not exist")
            layers += f"\n\t\tself.layer_{i} = {layer.construct(parameters, mode=self.type.lower())}\n\t\tself.layer_{i}_store = {layer.store()}"
        network += layers

        forward = "\n\tdef forward(self, input):"
        for i, inputs in enumerate(self.graph):
            in_ = ""
            out = ""
            for input_ in inputs:
                if input_ is None: in_ += f"input,"
                else:
                    layer, activation = input_
                    if activation is not None: out += f"\n\t\t{layer} = {activations[self.type.lower()][activation.lower()]}({layer})"
                    in_ += f"{layer},"
            forward += f"{out}\n\t\tout_{i} = self.layer_{i}({in_})\n\t\tself.layer_{i}_store(out_{i})"

        forward += f"\n\n\treturn out_{i}"
        network += forward

        train = "\n\n\ndef train():"
        #TODO: implement train constructor
        network += train

        test = "\n\n\ndef test():"
        #TODO: implement test constructor
        network += test

        self._overwrite(network)
        
    def _overwrite(self, code):
        """
        Overwrite the network's script with code implementation of the network

        Inputs
            :code: <str> containing the code
        """
        with open(self.interpreter_path, 'w') as script:
            script.seek(0)
            script.truncate(0)
            script.write(code)
            script.close()

    def train(self, dataset, weights, timeout: int = 60):
        """
        Trains the network
    
        Inputs
            :dataset: <Dataloader> to load the data for the network to train on
            :weights: <torch.Tensor> determining the network
            :timeout: <int> how long to run the code for before force stopping in seconds
    
        Throws
            <RunTimeError> if execution takes longer than timeout
        """
        raise NotImplementedError

    def test(self, input_: torch.Tensor, timeout: int = 60):
        """
        Trains the network
    
        Inputs
            :dataset: <Dataloader> to load the data for the network to train on
            :weights: <torch.Tensor> determining the network
            :timeout: <int> how long to run the code for before force stopping in seconds
    
        Throws
            <RunTimeError> if execution takes longer than timeout
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """ Override models.Model.__str__() """
        return f"{self.name}: <{self.type}> {len(json.loads(self.layers))} layers"