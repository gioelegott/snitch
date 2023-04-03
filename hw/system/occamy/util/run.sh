#!/bin/bash

if [ $# -ne 3 ]
then
    echo "Usage: $0 <benchmark_name> <dimension> <data_type>"
    exit 1
fi

timestamp=$(date +%s)


cd ./sw/device/apps/$1/
python generate_header.py $2 $3
cd ../../../../

make DEBUG=ON sw
bin/occamy_top.vsim sw/host/apps/$1/build/$1.elf
make traces

make BINARY=sw/device/apps/$1/build/$1.elf annotate -j



make logs/perf.csv
mv logs/perf.csv logs/history/perf_$1_$2_$3_$timestamp.csv

make logs/event.csv
mv logs/event.csv logs/history/event_$1_$2_$3_$timestamp.csv

echo -n "$1_$2_$3_$timestamp " >> results.txt
python ../../../util/read_csv.py logs/history/event_$1_$2_$3_$timestamp.csv >> results.txt