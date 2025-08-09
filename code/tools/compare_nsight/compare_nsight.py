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
            "memory_bandwidth": "8.0 TB/s",
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
#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import sqlite3
import argparse
import sys
import math
import subprocess
import os
import time
from tqdm import tqdm
import re

##########################################
# Helper functions.
##########################################
def safe_int(x):
    return int(x) if x is not None else 0

def safe_pct(left, right):
    """Return percentage change from left to right: ((right - left)/left * 100)."""
    left_val = safe_int(left)
    if left_val == 0:
        return 0.0
    return (safe_int(right) - left_val) / left_val * 100

def shorten_name(name, max_len=50):
    """Truncate a name to at most max_len characters."""
    s = str(name)
    if len(s) > max_len:
        return s[:max_len-3] + "..."
    return s

def unique_kernel_name(name, max_len=50):
    """
    Generate a unique, concise kernel name by:
      1. Removing template parameters (everything between '<' and '>').
      2. Splitting on '::' and taking the last two tokens.
      3. Removing all spaces.
      4. Capping the result to max_len characters.
    """
    name_no_templates = re.sub(r"<.*?>", "", str(name))
    parts = name_no_templates.split("::")
    if len(parts) >= 2:
        short_name = "::".join(parts[-2:]).strip()
    else:
        short_name = name_no_templates.strip()
    short_name = short_name.replace(" ", "")
    if len(short_name) > max_len:
        short_name = short_name[:max_len]
    return short_name

##########################################
# Prepare DB File.
##########################################
def prepare_db_file(db_file):
    base, ext = os.path.splitext(db_file)
    if ext.lower() == ".sqlite":
        return db_file
    out_file = base + ".sqlite"
    cmd = ["nsys", "stats", "-r", "cuda_api_sum", "--force-export=true", db_file, "-o", out_file]
    print(f"Converting raw nsys-prof file {db_file} to summary SQLite file {out_file} ...")
    with tqdm(total=0, bar_format="{desc}", dynamic_ncols=True) as pbar:
        pbar.set_description("Exporting to SQLite ...")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while proc.poll() is None:
            time.sleep(1)
        stdout, stderr = proc.communicate()
    return out_file

##########################################
# Custom aggregate functions.
##########################################
class Median:
    def __init__(self):
        self.values = []
    def step(self, value):
        if value is not None:
            self.values.append(value)
    def finalize(self):
        n = len(self.values)
        if n == 0:
            return None
        self.values.sort()
        mid = n // 2
        return self.values[mid] if n % 2 == 1 else (self.values[mid-1] + self.values[mid]) / 2.0

class Stdev:
    def __init__(self):
        self.count = 0; self.sum = 0.0; self.sum_sq = 0.0
    def step(self, value):
        if value is not None:
            self.count += 1; self.sum += value; self.sum_sq += value * value
    def finalize(self):
        if self.count < 2:
            return None
        mean = self.sum / self.count
        variance = (self.sum_sq - self.count * mean * mean) / (self.count - 1)
        return math.sqrt(variance)

class LowerQuartile:
    def __init__(self):
        self.values = []
    def step(self, value):
        if value is not None:
            self.values.append(value)
    def finalize(self):
        n = len(self.values)
        if n == 0:
            return None
        self.values.sort()
        lower_half = self.values[:n//2]
        if not lower_half:
            return self.values[0]
        m = len(lower_half); mid = m // 2
        return lower_half[mid] if m % 2 == 1 else (lower_half[mid-1] + lower_half[mid]) / 2.0

class UpperQuartile:
    def __init__(self):
        self.values = []
    def step(self, value):
        if value is not None:
            self.values.append(value)
    def finalize(self):
        n = len(self.values)
        if n == 0:
            return None
        self.values.sort()
        upper_half = self.values[n//2:] if n % 2 == 0 else self.values[n//2+1:]
        if not upper_half:
            return self.values[-1]
        m = len(upper_half); mid = m // 2
        return upper_half[mid] if m % 2 == 1 else (upper_half[mid-1] + upper_half[mid]) / 2.0

##########################################
# SQL Queries for Different Profiling Categories.
##########################################
# 1. Overall: capture duration.
OVERALL_QUERY = """
SELECT duration AS Duration_ns FROM ANALYSIS_DETAILS LIMIT 1;
"""

# 2. CUDA API events.
CUDA_API_QUERY = """
WITH
    summary AS (
        SELECT nameId,
               sum(end - start) AS total,
               count(*) AS num
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        GROUP BY nameId
    )
SELECT summary.num AS NumCalls,
       summary.total AS TotalTime_ns,
       CASE substr(ids.value, -6, 2)
            WHEN '_v' THEN substr(ids.value, 1, length(ids.value)-6)
            ELSE ids.value
       END AS Name
FROM summary
LEFT JOIN StringIds AS ids ON ids.id = summary.nameId
ORDER BY summary.total DESC;
"""

# 3. Kernel events.
KERNEL_QUERY_ALT = """
WITH
    summary AS (
        SELECT demangledName AS nameId,
               sum(end - start) AS total,
               count(*) AS num
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        GROUP BY demangledName
    )
SELECT summary.num AS KernelCalls,
       summary.total AS TotalKernelTime_ns,
       ids.value AS Name
FROM summary
LEFT JOIN StringIds AS ids ON ids.id = summary.nameId
ORDER BY summary.total DESC;
"""

# 4. Memcpy events.
MEMCPY_QUERY = """
WITH
    summary AS (
        SELECT copyKind AS kind,
               sum(bytes) AS total_bytes,
               count(*) AS num
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        GROUP BY copyKind
    )
SELECT summary.num AS num,
       summary.total_bytes AS total_bytes,
       CASE WHEN oper.label IS NOT NULL THEN oper.label ELSE summary.kind END AS Name
FROM summary
LEFT JOIN ENUM_CUDA_MEMCPY_OPER AS oper ON oper.id = summary.kind
ORDER BY summary.total_bytes DESC;
"""

# 5. NVTX_API events.
NVTX_API_QUERY = """
WITH summary AS (
    SELECT eventType,
           sum(end - start) AS TotalTime_ns,
           count(*) AS NumEvents
    FROM NVTX_EVENTS
    GROUP BY eventType
)
SELECT summary.NumEvents,
       summary.TotalTime_ns,
       CASE WHEN et.label IS NOT NULL THEN et.label ELSE summary.eventType END AS Name
FROM summary
LEFT JOIN ENUM_NSYS_EVENT_TYPE AS et ON et.id = summary.eventType
ORDER BY summary.TotalTime_ns DESC;
"""

# 6. NVTX_MARKERS: Fine-grained markers.
NVTX_MARKERS_QUERY = """
WITH summary AS (
    SELECT COALESCE(textId, eventType) AS marker,
           sum(end - start) AS TotalTime_ns,
           count(*) AS NumEvents
    FROM NVTX_EVENTS
    GROUP BY marker
)
SELECT summary.NumEvents,
       summary.TotalTime_ns,
       COALESCE(sid.value, summary.marker) AS Name
FROM summary
LEFT JOIN StringIds AS sid ON sid.id = summary.marker
ORDER BY summary.TotalTime_ns DESC;
"""

# 7. OS runtime API events.
OSRT_API_QUERY = """
WITH summary AS (
    SELECT nameId,
           sum(end - start) AS TotalTime_ns,
           count(*) AS NumCalls
    FROM OSRT_API
    GROUP BY nameId
)
SELECT summary.NumCalls,
       summary.TotalTime_ns,
       ids.value AS Name
FROM summary
LEFT JOIN StringIds AS ids ON ids.id = summary.nameId
ORDER BY summary.TotalTime_ns DESC;
"""

# 8. cuDNN events.
CUDNN_QUERY = """
WITH summary AS (
    SELECT nameId,
           sum(end - start) AS TotalTime_ns,
           count(*) AS NumCalls
    FROM CUDNN_EVENTS
    GROUP BY nameId
)
SELECT summary.NumCalls,
       summary.TotalTime_ns,
       ids.value AS Name
FROM summary
LEFT JOIN StringIds AS ids ON ids.id = summary.nameId
ORDER BY summary.TotalTime_ns DESC;
"""

# 9. Memset events.
MEMSET_QUERY = """
WITH summary AS (
    SELECT sum(end - start) AS TotalTime_ns,
           count(*) AS NumCalls
    FROM CUPTI_ACTIVITY_KIND_MEMSET
)
SELECT TotalTime_ns,
       NumCalls
FROM summary;
"""

# Dictionary mapping category names to queries.
# We'll change the order in run_comparisons() below.
QUERIES = {
    "Memcpy": MEMCPY_QUERY,
    "NVTX_API": NVTX_API_QUERY,
    "OSRT_API": OSRT_API_QUERY,
    "Memset": MEMSET_QUERY,
    "CUDNN": CUDNN_QUERY,
    "Kernel_Improvement": KERNEL_QUERY_ALT,
    "CUDA_API": CUDA_API_QUERY,
    "NVTX_MARKERS": NVTX_MARKERS_QUERY,
    "Overall": OVERALL_QUERY,
}

# Define key metric fields for each category.
METRIC_FIELDS = {
    "Overall": ["Duration_ns"],
    "CUDA_API": ["NumCalls", "TotalTime_ns"],
    "Kernel_Improvement": ["KernelCalls", "TotalKernelTime_ns"],
    "Memcpy": ["num", "total_bytes"],
    "NVTX_API": ["NumEvents", "TotalTime_ns"],
    "NVTX_MARKERS": ["NumEvents", "TotalTime_ns"],
    "OSRT_API": ["NumCalls", "TotalTime_ns"],
    "CUDNN": ["NumCalls", "TotalTime_ns"],
    "Memset": ["NumCalls", "TotalTime_ns"],
}

##########################################
# Comparison Class.
##########################################
class NsightCompare:
    def __init__(self, before_file, after_file):
        self.before_file = prepare_db_file(before_file)
        self.after_file = prepare_db_file(after_file)
        self.check_table("CUPTI_ACTIVITY_KIND_RUNTIME")

    def check_table(self, table_name):
        for db in [self.before_file, self.after_file]:
            try:
                conn = sqlite3.connect(db)
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
                if not cur.fetchone():
                    sys.exit(f"Error: {db} does not contain table {table_name}.")
                conn.close()
            except sqlite3.Error as e:
                sys.exit(f"Error connecting to {db}: {e}")

    def fetch_results(self, db_file, query, key_field="Name"):
        try:
            conn = sqlite3.connect(db_file)
        except sqlite3.Error as e:
            sys.exit(f"Error connecting to {db_file}: {e}")
        conn.create_aggregate("median", 1, Median)
        conn.create_aggregate("stdev", 1, Stdev)
        conn.create_aggregate("lower_quartile", 1, LowerQuartile)
        conn.create_aggregate("upper_quartile", 1, UpperQuartile)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        try:
            cur.execute(query)
        except sqlite3.Error as e:
            if "no such table" in str(e):
                return {}
            else:
                sys.exit(f"Error executing query on {db_file}: {e}")
        rows = cur.fetchall()
        conn.close()
        results = {}
        if not rows:
            return results
        actual_key_field = key_field if key_field in rows[0].keys() else list(rows[0].keys())[0]
        for row in rows:
            key = row[actual_key_field]
            results[key] = {col: row[col] for col in row.keys()}
        return results

    def compare_category(self, category, query, key_field="Name"):
        before_results = self.fetch_results(self.before_file, query, key_field)
        after_results = self.fetch_results(self.after_file, query, key_field)
        common_keys = set(before_results.keys()) & set(after_results.keys())
        if category == "Overall":
            if not before_results or not after_results:
                print(f"\n==== Overall Comparison for {category} ====")
                print("Overall capture duration not found.")
                return
            b = list(before_results.values())[0]
            a = list(after_results.values())[0]
            field = "Duration_ns"
            left_val = safe_int(b.get(field, 0))
            right_val = safe_int(a.get(field, 0))
            delta = right_val - left_val
            pct = (delta / left_val * 100) if left_val else 0
            print(f"\n==== Overall Comparison for {category} ====")
            print(f"{'Overall Duration_ns':<50s} {left_val:15d} {right_val:15d} {delta:15d} {pct:15.1f}%")
            return

        if category == "Memset":
            if not before_results or not after_results:
                return
            b = list(before_results.values())[0]
            a = list(after_results.values())[0]
            print(f"\n==== Comparison for {category} ====")
            print("{:<50s} {:>15s} {:>15s} {:>15s} {:>15s}".format("Metric", "Left", "Right", "Delta", "PctDelta"))
            for metric in METRIC_FIELDS[category]:
                b_val = safe_int(b.get(metric, 0))
                a_val = safe_int(a.get(metric, 0))
                delta = a_val - b_val
                pct = (delta / b_val * 100) if b_val else 0
                print("{:<50s} {:15d} {:15d} {:15d} {pct:15.1f}%".format(metric, b_val, a_val, delta, pct=pct))
            return

        if category == "Kernel_Improvement":
            header = (f"{'Name':<50s} "
                      f"{'Left_KernelCalls':>15s} {'Right_KernelCalls':>15s} {'Delta_KernelCalls':>15s} {'PctDelta_KernelCalls':>15s} "
                      f"{'Left_TotalKernelTime_ns':>20s} {'Right_TotalKernelTime_ns':>20s} {'Delta_TotalKernelTime_ns':>20s} {'PctDelta_TotalKernelTime_ns':>15s}")
            print(f"\n==== Comparison for {category} ====")
            print(header)
            sorted_keys = sorted(common_keys, key=lambda k: safe_pct(before_results[k].get("TotalKernelTime_ns", 0),
                                                                      after_results[k].get("TotalKernelTime_ns", 0)))
            for key in sorted_keys:
                b = before_results[key]
                a = after_results[key]
                left_calls = safe_int(b.get("KernelCalls", 0))
                right_calls = safe_int(a.get("KernelCalls", 0))
                delta_calls = right_calls - left_calls
                pct_calls = (delta_calls / left_calls * 100) if left_calls else 0

                left_total = safe_int(b.get("TotalKernelTime_ns", 0))
                right_total = safe_int(a.get("TotalKernelTime_ns", 0))
                delta_total = right_total - left_total
                pct_total = (delta_total / left_total * 100) if left_total else 0

                print(f"{unique_kernel_name(key):<50s} {left_calls:15d} {right_calls:15d} {delta_calls:15d} {pct_calls:15.1f}% "
                      f"{left_total:20d} {right_total:20d} {delta_total:20d} {pct_total:15.1f}%")
            return

        # For all other categories.
        metric_fields = METRIC_FIELDS.get(category, [])
        header = f"\n==== Comparison for {category} ====\n"
        header += f"{'Name':<50s}"
        for field in metric_fields:
            header += f" {('Left_' + field):>15s} {('Right_' + field):>15s} {('Delta_' + field):>15s} {('PctDelta_' + field):>15s}"
        print(header)
        sorted_keys = sorted(common_keys, key=lambda k: safe_pct(before_results[k].get(metric_fields[1], 0),
                                                                  after_results[k].get(metric_fields[1], 0)))
        for key in sorted_keys:
            b = before_results[key]
            a = after_results[key]
            line = f"{unique_kernel_name(key):<50s}"
            for field in metric_fields:
                b_val = safe_int(b.get(field, 0))
                a_val = safe_int(a.get(field, 0))
                delta = a_val - b_val
                pct = (delta / b_val * 100) if b_val else 0
                line += f" {b_val:15d} {a_val:15d} {delta:15d} {pct:15.1f}%"
            print(line)

    def run_comparisons(self):
        # New order:
        # 1. Memcpy
        # 2. NVTX_API
        # 3. OSRT_API
        # 4. Memset
        # 5. CUDNN
        # 6. Kernel_Improvement
        # 7. CUDA_API
        # 8. NVTX_MARKERS
        # 9. Overall
        self.compare_category("Memcpy", QUERIES["Memcpy"], key_field="Name")
        self.compare_category("NVTX_API", QUERIES["NVTX_API"], key_field="Name")
        self.compare_category("OSRT_API", QUERIES["OSRT_API"], key_field="Name")
        self.compare_category("Memset", QUERIES["Memset"], key_field="TotalTime_ns")
        self.compare_category("CUDNN", QUERIES["CUDNN"], key_field="Name")
        self.compare_category("Kernel_Improvement", QUERIES["Kernel_Improvement"], key_field="Name")
        self.compare_category("CUDA_API", QUERIES["CUDA_API"], key_field="Name")
        self.compare_category("NVTX_MARKERS", QUERIES["NVTX_MARKERS"], key_field="Name")
        self.compare_category("Overall", QUERIES["Overall"], key_field="Duration_ns")

##########################################
# Main Entry Point.
##########################################
def main():
    parser = argparse.ArgumentParser(
        description="Compare Nsight Systems profiling metrics between two .nsys-prof or .sqlite files."
    )
    parser.add_argument("before", help="Path to the left .nsys-prof or .sqlite file")
    parser.add_argument("after", help="Path to the right .nsys-prof or .sqlite file")
    args = parser.parse_args()

    # Check file existence.
    if not os.path.exists(args.before):
        sys.exit(f"Error: The file '{args.before}' does not exist.")
    if not os.path.exists(args.after):
        sys.exit(f"Error: The file '{args.after}' does not exist.")

    comparer = NsightCompare(args.before, args.after)
    comparer.run_comparisons()

if __name__ == "__main__":
    main()

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
