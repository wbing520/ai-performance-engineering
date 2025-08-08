import torch.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch.cuda.nvtx as nvtx
import torch
import os

def get_architecture():
    """Detect and return the current GPU architecture."""
    if not torch.cuda.is_available():
        return "cpu"
    
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    
    # Architecture detection
    if compute_capability == "9.0":
        return "hopper"  # H100/H200
    elif compute_capability == "10.0":
        return "blackwell"  # B200/B300
    else:
        return "other"

def get_architecture_info():
    """Get detailed architecture information."""
    arch = get_architecture()
    if arch == "hopper":
        return {
            "name": "Hopper H100/H200",
            "compute_capability": "9.0",
            "sm_version": "sm_90",
            "memory_bandwidth": "3.35 TB/s",
            "tensor_cores": "4th Gen",
            "features": ["HBM3", "Transformer Engine", "Dynamic Programming"]
        }
    elif arch == "blackwell":
        return {
            "name": "Blackwell B200/B300",
            "compute_capability": "10.0",
            "sm_version": "sm_100",
            "memory_bandwidth": "3.2 TB/s",
            "tensor_cores": "4th Gen",
            "features": ["HBM3e", "TMA", "NVLink-C2C"]
        }
    else:
        return {
            "name": "Other",
            "compute_capability": "Unknown",
            "sm_version": "Unknown",
            "memory_bandwidth": "Unknown",
            "tensor_cores": "Unknown",
            "features": []
        }
## Relevant for supplying routes
from fastapi import FastAPI

## Relevant for supporting docker utilities
import subprocess
import docker

## Relevant for taking in NVAPI_Key
import os
from pydantic import BaseModel, Field, validator
from typing import Any, Dict

client = docker.from_env()
app = FastAPI()

@app.get("/")
async def read_root():
    """Default Route: Usually good to render instructions..."""
    return {"Hello": "World"}


@app.get("/help")
async def read_root():
    """Typical help route to tell users what they can do"""
    return {"Options": "[/containers, /containers/{container_id}/logs, containers/{container_id}/restart]"}


@app.get("/healthy")
async def read_root():
    """Typical health check route. Can be used by other microservices to tell whether they can rely on this one"""
    return True


@app.get("/containers")
async def list_containers():
    """A listing route. Lists current set of containers"""
    containers = client.containers.list(all=True)
    return [{"id": container.id, "name": container.name, "status": container.status} for container in containers]


@app.get("/containers/{container_name}/logs")
async def get_container_logs(container_name: str):
    """Route that allows you to query the log file of the container"""
    try:
        container = client.containers.get(container_name)
        logs = container.logs()
        return {"logs": logs.decode('utf-8')}
    except NotFound:
        return {"error": f"Container `{container_name}` not found"}


@app.post("/containers/{container_name}/restart")
async def restart_container(container_name: str):
    """Just in case it ever becomes necessary (probably won't be though)"""
    try:
        container = client.containers.get(container_name)
        container.restart()
        return {"status": f"Container {container_name} restarted successfully"}
    except NotFound:
        return {"error": f"Container {container_name} not found"}

######################################################################################
## More info: https://fastapi.tiangolo.com/tutorial/body/#import-pydantics-basemodel
class Key(BaseModel):

    ## Possible variables that your message body expects
    nvapi_key: str

    # Validator using custom function
    @validator('nvapi_key')
    def check_nvapi_prefix_function(cls, v):
        if not v.startswith('nvapi-'):
            raise ValueError('nvapi_key must start with "nvapi-"')
        return v

API_KEY = None

@app.post("/set_key/")
async def set_key(key: Key):
    global API_KEY 
    API_KEY = key.nvapi_key 
    return {"result" : "Key set successfully"}

@app.get("/get_key/")
async def get_key():
    global API_KEY
    return {"nvapi_key": API_KEY}
# Architecture-specific optimizations
if torch.cuda.is_available():
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    
    if compute_capability == "9.0":  # Hopper H100/H200
        torch._inductor.config.triton.use_hopper_optimizations = True
        torch._inductor.config.triton.hbm3_optimizations = True
    elif compute_capability == "10.0":  # Blackwell B200/B300
        torch._inductor.config.triton.use_blackwell_optimizations = True
        torch._inductor.config.triton.hbm3e_optimizations = True
        torch._inductor.config.triton.tma_support = True
    
    # Enable latest PyTorch 2.8 features
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.triton.autotune_mode = "max-autotune"
    torch._dynamo.config.automatic_dynamic_shapes = True
