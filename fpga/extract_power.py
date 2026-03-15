import argparse
import re
import sys

def parse_power_report(path):
    data = {}
    with open(path, 'r') as f:
        for line in f:
            m = re.search(r'Total On-Chip Power.*?:\s+([\d.]+)', line)
            if m:
                data['total_power_w'] = float(m.group(1))

            m = re.search(r'Dynamic.*?:\s+([\d.]+)', line)
            if m and 'dynamic_power_w' not in data:
                data['dynamic_power_w'] = float(m.group(1))

            m = re.search(r'Device Static.*?:\s+([\d.]+)', line)
            if m:
                data['static_power_w'] = float(m.group(1))

            m = re.search(r'Block RAM\s*:\s+([\d.]+)', line)
            if m:
                data['bram_power_w'] = float(m.group(1))

            m = re.search(r'Clocks\s*:\s+([\d.]+)', line)
            if m:
                data['clock_power_w'] = float(m.group(1))

            m = re.search(r'Logic\s*:\s+([\d.]+)', line)
            if m and 'logic_power_w' not in data:
                data['logic_power_w'] = float(m.group(1))

    return data

def parse_utilization_report(path):
    data = {}
    with open(path, 'r') as f:
        content = f.read()

    m = re.search(r'Slice LUTs\*?\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)', content)
    if m:
        data['luts_used'] = int(m.group(1).replace(',', ''))
        data['luts_total'] = int(m.group(2).replace(',', ''))

    m = re.search(r'(?:Slice Registers|Register as Flip Flop)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)', content)
    if m:
        data['ffs_used'] = int(m.group(1).replace(',', ''))
        data['ffs_total'] = int(m.group(2).replace(',', ''))

    m = re.search(r'Block RAM Tile\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)', content)
    if m:
        data['bram_used'] = float(m.group(1))
        data['bram_total'] = float(m.group(2))

    m = re.search(r'DSPs?\s*\|\s*([\d]+)\s*\|\s*([\d]+)', content)
    if m:
        data['dsps_used'] = int(m.group(1))
        data['dsps_total'] = int(m.group(2))

    return data

def manual_entry():
    return {
        'target': 'Xilinx VU47P (xcvu47p, AWS F2)',
        'cores': 16,
        'neurons_per_core': 1024,
        'total_neurons': 16384,
        'clock_mhz': 62.5,
        'bram36k_used': 1999,
        'bram36k_total': 3576,
        'bram_pct': 55.9,
        'wns_ns': 0.003,
        'throughput_ts_per_sec': 8690,
        'asic_estimate_note': 'FPGA power / 10-20x for ASIC estimate',
    }

def print_paper_table(power, util, manual):
    print("\nRESOURCE UTILIZATION")
    print(f"Target:          {manual['target']}")
    print(f"Cores:           {manual['cores']}")
    print(f"Neurons:         {manual['total_neurons']:,}")
    print(f"Clock:           {manual['clock_mhz']} MHz")
    print(f"WNS:             +{manual['wns_ns']} ns (timing MET)")
    print(f"BRAM36K:         {manual['bram36k_used']} / {manual['bram36k_total']} "
          f"({manual['bram_pct']:.1f}%)")

    if util:
        if 'luts_used' in util:
            lut_pct = 100 * util['luts_used'] / util['luts_total']
            print(f"LUTs:            {util['luts_used']:,} / {util['luts_total']:,} "
                  f"({lut_pct:.1f}%)")
        if 'ffs_used' in util:
            ff_pct = 100 * util['ffs_used'] / util['ffs_total']
            print(f"Flip-Flops:      {util['ffs_used']:,} / {util['ffs_total']:,} "
                  f"({ff_pct:.1f}%)")
        if 'dsps_used' in util:
            print(f"DSPs:            {util['dsps_used']} / {util['dsps_total']}")

    print(f"\nThroughput:      {manual['throughput_ts_per_sec']:,} timesteps/sec")

    if power:
        print(f"\nPOWER (from Vivado report_power)")
        for k, v in sorted(power.items()):
            print(f"  {k}: {v:.3f} W")

        if 'dynamic_power_w' in power:
            asic_lo = power['dynamic_power_w'] / 20
            asic_hi = power['dynamic_power_w'] / 10
            print(f"\nASIC estimate: {asic_lo*1000:.0f} - {asic_hi*1000:.0f} mW "
                  f"(FPGA dynamic / 10-20x)")

def main():
    parser = argparse.ArgumentParser(description="Extract Vivado power/utilization")
    parser.add_argument("power_report", nargs='?', help="Vivado power report file")
    parser.add_argument("util_report", nargs='?', help="Vivado utilization report file")
    parser.add_argument("--manual", action="store_true",
                        help="Use known F2 build numbers")
    args = parser.parse_args()

    manual = manual_entry()
    power = {}
    util = {}

    if args.power_report:
        power = parse_power_report(args.power_report)
    if args.util_report:
        util = parse_utilization_report(args.util_report)

    print_paper_table(power, util, manual)

if __name__ == "__main__":
    main()
