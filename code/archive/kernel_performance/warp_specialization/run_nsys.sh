#!/usr/bin/env bash
nsys profile -o nsys_warp_spec add_specialized
nsys stats --report summary,cuda_api --format sqlite,csv nsys_warp_spec -o nsys_warp_spec