"""
generate tensor comprehension base line
"""

import time
import argparse
import logging
import sys
import os
import re

import numpy as np
import torch

import tensor_comprehensions as tc

#tc.GlobalDebugInit(["--dump_cuda=true"])

resnet_wkls = [
    # resnet18
    (1, 224, 3,   64,  7, 2, 3, "float32"),
    (1, 56,  64,  64,  3, 1, 1, "float32"),
    (1, 56,  64,  64,  1, 1, 0, "float32"),
    (1, 56,  64,  128, 3, 2, 1, "float32"),
    (1, 56,  64,  128, 1, 2, 0, "float32"),
    (1, 28,  128, 128, 3, 1, 1, "float32"),
    (1, 28,  128, 256, 3, 2, 1, "float32"),
    (1, 28,  128, 256, 1, 2, 0, "float32"),
    (1, 14,  256, 256, 3, 1, 1, "float32"),
    (1, 14,  256, 512, 3, 2, 1, "float32"),
    (1, 14,  256, 512, 1, 2, 0, "float32"),
    (1, 7,   512, 512, 3, 1, 1, "float32"),
]

settings = {
    "threads": 8, "generations": 8, "pop_size": 50, "number_elites": 2
}

conv2d_lang = """
def conv2d(float(N, C, H, W) I, float(M, C, KH, KW) W1) -> (output) {{
    output(n, m, h, w) +=! I(n, c, {sh} * h + kh, {sw} * w + kw) * W1(m, c, kh, kw)
}}
"""

def name2workloads(name):
    if name == 'resnet':
        return resnet_wkls
    else:
        raise RuntimeError("Invalid task " + name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--n-ave-curve", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if '-' in args.task:
        # subprocess called by master process
        # this mode is used to extract the stdout of c function by python script
        task, no = args.task.split('-')
        wkls = name2workloads(task)

        wkl = wkls[int(no)]

        N, H, C, M, KH, sh, pad, dtype = wkl
        H = H + 2 * pad
        W = H
        KW = KH
        sw = sh

        cache_file = 'cache_%s.tc' % str(wkl)
        conv2d = tc.define(conv2d_lang, name="conv2d", constants={"sh": sh, "sw": sw})

        options = tc.Options('conv')

        data, kernel = torch.randn(N, C, H, W).cuda(), torch.randn(M, C, KH, KW).cuda()
        conv2d.autotune(data, kernel, options=options, **settings)

        out = conv2d(data, kernel)
    else:
        # master mode
        tmp_filename = 'tc_output.log'
        if os.path.isfile(tmp_filename):
            os.remove(tmp_filename)

        wkls = name2workloads(args.task)

        results = []

        for i in range(len(wkls)):
            N, H, C, M, KH, sh, pad, dtype = wkls[i]
            KW = KH
            sw = sh
            OH = (H + 2 * pad - KH) // sh + 1
            OW = OH

            costs = []
            gflops = []
            for j in range(args.n_ave_curve):
                flop = 2 * N * OH * OW * C * M * KH * KW
                cmd = 'python3 tc_baseline.py --task "%s-%d" >> %s 2>> %s' \
                        % (args.task, i, tmp_filename, tmp_filename)
                print(cmd)
                os.system(cmd)

                with open(tmp_filename) as f:
                    lines = list(f.readlines())
                    # trackback to extract best for every generation
                    best = 1e9
                    for line in reversed(lines):
                        if 'Generation' in line:
                            find = re.search("us:\s+(\d+)", line)
                            if find is not None:
                                cost = int(find.group(1))
                                best = min(best, cost)
                        if 'Autotuning' in line:
                            break
                    cost = best / 1e6 # us -> s
                    gflop = (flop / cost / 1e9)

                    costs.append([cost])
                    gflops.append([gflop])

            print("cost: %s\tgflops: %s" % (costs, gflops))
            results.append(("%s-%d" % (args.task, i+1), "%s" % costs, "%s" % gflops))

        import tvm
        device_name = str(tvm.gpu(0).device_name).replace(' ', '-')
        outfile_name = 'tc-baseline-%s.tsv' % device_name
        with open(outfile_name, 'w') as fout:
            for res in results:
                fout.write("\t".join(res) + '\n')

