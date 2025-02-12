for i in ./scripts_custom_classification/scripts_all/MambaSL_CLS_*.sh 
do
    echo "Running $i"
    echo "Result will be saved in ./scripts_custom_classification/results/${$(basename $i)%.*}.out"
    nohup bash $i > ./scripts_custom_classification/results/${$(basename $i)%.*}.out &
done