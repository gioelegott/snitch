if [ $# -ne 4 ]
then
    echo "Usage: $0 <benchmark_name> <dimension_start> <dimension_end> <data_type>"
    exit 1
fi

for i in $(seq $2 1 $3)
do
   ../../../util/run.sh $1 $i $4
done