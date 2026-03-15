#!/bin/bash
# ============================================================================
# F2 Deploy Script — Build + Deploy Neuromorphic Chip to AWS F2
# ============================================================================
#
# Prerequisites:
#   1. AWS FPGA HDK cloned and set up:
#      git clone https://github.com/aws/aws-fpga
#      cd aws-fpga && source hdk_setup.sh
#
#   2. This repository cloned at $NEURO_DIR:
#      export NEURO_DIR=/path/to/neuromorphic-chip
#
#   3. S3 bucket for AFI artifacts:
#      export AFI_BUCKET=my-fpga-bucket
#      export AFI_PREFIX=neuromorphic-v2.3
#
# Usage:
#   ./deploy_f2.sh [--build-only | --load-only | --test]
# ============================================================================

set -euo pipefail

NEURO_DIR="${NEURO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
AFI_BUCKET="${AFI_BUCKET:-}"
AFI_PREFIX="${AFI_PREFIX:-neuromorphic-v2.3}"
CL_DIR="${CL_DIR:-$HDK_DIR/cl/developer_designs/cl_neuromorphic}"
MODE="${1:---full}"

echo "============================================"
echo "  Neuromorphic Chip v2.3 — F2 Deployment"
echo "============================================"
echo "  NEURO_DIR: $NEURO_DIR"
echo "  CL_DIR:    $CL_DIR"
echo "  Mode:      $MODE"
echo ""

# ---- Step 1: Copy design files into HDK CL tree ----
copy_design() {
    echo "--- Copying design files ---"
    mkdir -p "$CL_DIR/design"
    mkdir -p "$CL_DIR/build/constraints"

    # CL wrapper + bridge
    cp "$NEURO_DIR/fpga/f2/cl_neuromorphic.v"          "$CL_DIR/design/"
    cp "$NEURO_DIR/fpga/f2/cl_neuromorphic_defines.vh"  "$CL_DIR/design/"
    cp "$NEURO_DIR/rtl/axi_uart_bridge.v"               "$CL_DIR/design/"

    # Neuromorphic RTL (excluding UART modules — BYPASS_UART=1)
    for f in sram.v spike_fifo.v scalable_core_v2.v neuromorphic_mesh.v \
             async_noc_mesh.v async_router.v sync_tree.v chip_link.v \
             host_interface.v neuromorphic_top.v rv32i_core.v \
             rv32im_cluster.v mmio_bridge.v multi_chip_router.v; do
        cp "$NEURO_DIR/rtl/$f" "$CL_DIR/design/"
    done

    # Constraints
    cp "$NEURO_DIR/fpga/f2/cl_synth_user.xdc"   "$CL_DIR/build/constraints/"
    cp "$NEURO_DIR/fpga/f2/cl_timing_user.xdc"   "$CL_DIR/build/constraints/"

    # Build source list
    cp "$NEURO_DIR/fpga/f2/build_f2.tcl" "$CL_DIR/build/scripts/cl_build_user.tcl"

    echo "  Copied $(ls "$CL_DIR/design/"*.v 2>/dev/null | wc -l) Verilog files"
}

# ---- Step 2: Build DCP (synthesis + implementation) ----
build_dcp() {
    echo ""
    echo "--- Building DCP (this takes 4-8 hours) ---"
    cd "$CL_DIR/build/scripts"
    ./aws_build_dcp_from_cl.sh -clock_recipe_a A1  # A1 = 250 MHz
    echo "  DCP build complete"

    # Check for timing failures
    local timing_rpt="$CL_DIR/build/checkpoints/to_aws/*.SH_CL_routed.rpt"
    if grep -q "VIOLATED" $timing_rpt 2>/dev/null; then
        echo "  WARNING: Timing violations detected! Check reports."
    else
        echo "  Timing met at 250 MHz"
    fi
}

# ---- Step 3: Create AFI ----
create_afi() {
    if [ -z "$AFI_BUCKET" ]; then
        echo "  ERROR: Set AFI_BUCKET environment variable"
        exit 1
    fi

    echo ""
    echo "--- Creating AFI ---"
    local tar_file=$(ls "$CL_DIR/build/checkpoints/to_aws/"*.tar 2>/dev/null | head -1)
    if [ -z "$tar_file" ]; then
        echo "  ERROR: No .tar file found in checkpoints/to_aws/"
        exit 1
    fi

    aws s3 cp "$tar_file" "s3://$AFI_BUCKET/$AFI_PREFIX/"

    local tar_name=$(basename "$tar_file")
    aws ec2 create-fpga-image \
        --name "neuromorphic-v2.3-16core" \
        --description "Neuromorphic chip v2.3, 16 cores x 1024 neurons, F2 VU47P" \
        --input-storage-location "Bucket=$AFI_BUCKET,Key=$AFI_PREFIX/$tar_name" \
        --logs-storage-location "Bucket=$AFI_BUCKET,Key=$AFI_PREFIX/logs/" \
        | tee /tmp/afi_create_output.json

    echo ""
    echo "  AFI creation submitted. Monitor with:"
    echo "    aws ec2 describe-fpga-images --fpga-image-ids <afi-id>"
}

# ---- Step 4: Load AFI ----
load_afi() {
    local afi_id="${AFI_ID:-}"
    if [ -z "$afi_id" ]; then
        echo "  ERROR: Set AFI_ID environment variable (e.g., afi-XXXXXXXX)"
        exit 1
    fi

    local agfi_id="${AGFI_ID:-}"
    if [ -z "$agfi_id" ]; then
        echo "  ERROR: Set AGFI_ID environment variable (e.g., agfi-XXXXXXXX)"
        exit 1
    fi

    echo ""
    echo "--- Loading AFI onto slot 0 ---"
    sudo fpga-load-local-image -S 0 -I "$agfi_id"
    sleep 2
    sudo fpga-describe-local-image -S 0 -H
    echo "  AFI loaded"
}

# ---- Step 5: Run test ----
run_test() {
    echo ""
    echo "--- Running connectivity test ---"
    python3 "$NEURO_DIR/fpga/f2_host.py" --test-loopback
    echo ""
    echo "--- Running spike test ---"
    python3 "$NEURO_DIR/fpga/f2_host.py" --test-spike
}

# ---- Main ----
case "$MODE" in
    --build-only)
        copy_design
        build_dcp
        ;;
    --afi-only)
        create_afi
        ;;
    --load-only)
        load_afi
        ;;
    --test)
        run_test
        ;;
    --full)
        copy_design
        build_dcp
        create_afi
        echo ""
        echo "============================================"
        echo "  BUILD COMPLETE"
        echo "============================================"
        echo "  Next steps:"
        echo "    1. Wait for AFI to become available"
        echo "    2. export AFI_ID=afi-XXXXXXXX"
        echo "    3. export AGFI_ID=agfi-XXXXXXXX"
        echo "    4. ./deploy_f2.sh --load-only"
        echo "    5. ./deploy_f2.sh --test"
        echo "============================================"
        ;;
    *)
        echo "Usage: $0 [--build-only | --afi-only | --load-only | --test | --full]"
        exit 1
        ;;
esac
