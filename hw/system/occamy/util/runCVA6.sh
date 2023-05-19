#!/bin/bash

if [ $# -ne 3 ]
then
    echo "Usage: $0 <benchmark_name> <dimension> <data_type>"
    exit 1
fi

timestamp=$(date +%s)

cd sw/host/apps/$1CVA6/
python generate_header.py $2 $3
cd ../../../../

rm logs/*

make DEBUG=ON sw

tstart=$(date +%s)
bin/occamy_top.vsim sw/host/apps/$1CVA6/build/$1CVA6.elf
tend=$(date +%s)


make traces

make BINARY=sw/host/apps/$1CVA6/build/$1CVA6.elf annotate

make logs/perf.csv

make logs/event.csv

#mkdir logs/history/$1CVA6_$2_$3_$timestamp


#find ./logs -maxdepth 1 -type f | xargs cp -t ./logs/history/$1CVA6_$2_$3_$timestamp


echo -n "$1_$2_$3_$timestamp " >> logs/history/history.txt
#python util/read_csv_CVA6.py logs/history/$1CVA6_$2_$3_$timestamp/event.csv >> logs/history/history.txt
python util/read_csv_CVA6.py logs/event.csv >> logs/history/history.txt

echo -n "$1_$2_$3_$timestamp =$tend-$tstart " >> logs/history/historyCVA6.txt
#python util/read_csv_CVA6.py logs/history/$1CVA6_$2_$3_$timestamp/event.csv >> logs/history/historyCVA6.txt
python util/read_csv_CVA6.py logs/event.csv >> logs/history/historyCVA6.txt

