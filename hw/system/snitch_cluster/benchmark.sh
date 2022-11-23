# Benchmark shell script

for dim in 5 10 20 30 40 50
do

mkdir "/scratch2/mbertuletti/snitch/results_softmax_2/dim_$dim"

for i in 1 2 3 4 5 6 7 8 9 10
do

echo "Iteration $i"
cd /scratch2/mbertuletti/snitch/sw/dphpc/data
python3 data_gen_softmax_csr.py --dimension $dim
cd /scratch2/mbertuletti/snitch/hw/system/snitch_cluster
make softmax_csr -C sw/build/dphpc
./bin/snitch_cluster.vlt ./sw/build/dphpc/softmax_csr
make traces
mv /scratch2/mbertuletti/snitch/hw/system/snitch_cluster/logs "/scratch2/mbertuletti/snitch/results_softmax_2/dim_$dim/logs$i"

done

done
