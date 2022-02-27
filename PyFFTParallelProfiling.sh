#!/bin/bash
END=12
for i in $(seq 1 $END); do ./build.x86_64-conda_cos6-gcc93-o2g/run PyFFTParallel -l 40 -p 40 -b $i; done

exit 0