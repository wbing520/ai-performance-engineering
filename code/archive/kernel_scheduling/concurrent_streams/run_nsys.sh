#!/bin/bash
nsys profile --trace=cuda --output concurrent_report ./concurrent_stream_kernels
