if [ $# -ne 4 ]
then
    echo "Usage: $0 <benchmark_name> <dimension_start> <dimension_end> <data_type>"
    exit 1
fi

for i in $(seq $2 4 $3)
do
   ./util/runCVA6.sh $1 $i $4
done