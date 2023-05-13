#!/bin/bash

if [ $# -ne 3 ] && [ $# -ne 4 ]
then
    echo "Usage: $0 <benchmark_name> <dimension> <data_type> [-v]"
    exit 1
fi

timestamp=$(date +%s)

#generating data
cd sw/host/apps/$1/
python generate_header.py $2 $3
cd ../../../../

rm logs/*

make DEBUG=ON sw


if echo $* | grep -e "-v" -q
then
    python ../../ip/test/src/verification.py bin/occamy_top.vsim sw/host/apps/$1/build/$1.elf
else

    bin/occamy_top.vsim sw/host/apps/$1/build/$1.elf

    make traces

    make BINARY=sw/device/apps/$1/build/$1.elf annotate -j

    make logs/perf.csv

    make logs/event.csv

    ../../../util/trace/layout_events.py logs/event.csv sw/host/layout.csv -o logs/trace.csv
    ../../../util/trace/eventvis.py -o logs/trace.json logs/trace.csv

    mkdir logs/history/$1_$2_$3_$timestamp

    find ./logs -maxdepth 1 -type f | xargs cp -t ./logs/history/$1_$2_$3_$timestamp

    echo -n "$1_$2_$3_$timestamp " >> logs/history/history.txt
    python util/read_csv_SnitchCluster.py logs/history/$1_$2_$3_$timestamp/event.csv >> logs/history/history.txt

    echo -n "$1_$2_$3_$timestamp " >> logs/history/historySnitchCluster.txt
    python util/read_csv_SnitchCluster.py logs/history/$1_$2_$3_$timestamp/event.csv -x >> logs/history/historySnitchCluster.txt
fi