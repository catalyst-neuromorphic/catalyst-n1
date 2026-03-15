#!/bin/bash
set -euo pipefail

NEURO_DIR="${NEURO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
AFI_BUCKET="${AFI_BUCKET:-}"
AFI_PREFIX="${AFI_PREFIX:-neuromorphic}"
CL_DIR="${CL_DIR:-$HDK_DIR/cl/developer_designs/cl_neuromorphic}"
MODE="${1:---full}"

copy_design() {
    mkdir -p "$CL_DIR/design"
    mkdir -p "$CL_DIR/build/constraints"

    cp "$NEURO_DIR/fpga/f2/cl_neuromorphic.sv"         "$CL_DIR/design/"
    cp "$NEURO_DIR/fpga/f2/cl_neuromorphic_defines.vh"  "$CL_DIR/design/"
    cp "$NEURO_DIR/rtl/axi_uart_bridge.v"               "$CL_DIR/design/"

    for f in sram.v spike_fifo.v async_fifo.v scalable_core_v2.v neuromorphic_mesh.v \
             async_noc_mesh.v async_router.v chip_link.v \
             host_interface.v neuromorphic_top.v rv32i_core.v \
             rv32im_cluster.v multi_chip_router.v; do
        cp "$NEURO_DIR/rtl/$f" "$CL_DIR/design/"
    done

    cp "$NEURO_DIR/fpga/f2/cl_synth_user.xdc"   "$CL_DIR/build/constraints/"
    cp "$NEURO_DIR/fpga/f2/cl_timing_user.xdc"   "$CL_DIR/build/constraints/"
    cp "$NEURO_DIR/fpga/f2/build_f2.tcl" "$CL_DIR/build/scripts/cl_build_user.tcl"
}

build_dcp() {
    cd "$CL_DIR/build/scripts"
    ./aws_build_dcp_from_cl.sh -clock_recipe_a A1
}

create_afi() {
    if [ -z "$AFI_BUCKET" ]; then
        echo "Set AFI_BUCKET environment variable"
        exit 1
    fi

    local tar_file=$(ls "$CL_DIR/build/checkpoints/to_aws/"*.tar 2>/dev/null | head -1)
    if [ -z "$tar_file" ]; then
        echo "No .tar file found in checkpoints/to_aws/"
        exit 1
    fi

    aws s3 cp "$tar_file" "s3://$AFI_BUCKET/$AFI_PREFIX/"

    local tar_name=$(basename "$tar_file")
    aws ec2 create-fpga-image \
        --name "catalyst-n1" \
        --description "Catalyst N1, 16 cores x 1024 neurons, F2 VU47P" \
        --input-storage-location "Bucket=$AFI_BUCKET,Key=$AFI_PREFIX/$tar_name" \
        --logs-storage-location "Bucket=$AFI_BUCKET,Key=$AFI_PREFIX/logs/" \
        | tee /tmp/afi_create_output.json
}

load_afi() {
    local agfi_id="${AGFI_ID:-}"
    if [ -z "$agfi_id" ]; then
        echo "Set AGFI_ID environment variable"
        exit 1
    fi
    sudo fpga-load-local-image -S 0 -I "$agfi_id"
    sleep 2
    sudo fpga-describe-local-image -S 0 -H
}

run_test() {
    python3 "$NEURO_DIR/fpga/f2_host.py" --test-loopback
    python3 "$NEURO_DIR/fpga/f2_host.py" --test-spike
}

case "$MODE" in
    --build-only) copy_design; build_dcp ;;
    --afi-only)   create_afi ;;
    --load-only)  load_afi ;;
    --test)       run_test ;;
    --full)       copy_design; build_dcp; create_afi ;;
    *)            echo "Usage: $0 [--build-only | --afi-only | --load-only | --test | --full]"; exit 1 ;;
esac
