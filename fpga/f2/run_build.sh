#!/bin/bash
set -e
source /opt/Xilinx/2025.2/Vivado/settings64.sh
cd /home/ubuntu/aws-fpga
source hdk_setup.sh
export CL_DIR=/home/ubuntu/aws-fpga/hdk/cl/developer_designs/cl_neuromorphic
echo "Starting build at $(date)"
cd /home/ubuntu/aws-fpga/hdk/cl/developer_designs/cl_neuromorphic/build/scripts
python3 aws_build_dcp_from_cl.py -c cl_neuromorphic --no-encrypt
echo "Build finished at $(date)"
