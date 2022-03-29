#!/bin/bash
END=10
for i in $(seq 1 $END); do time ./build.x86_64-conda_cos6-gcc93-o2g/run PyFFTParallel -l 40 -p 10 -b $i; done

exit 0