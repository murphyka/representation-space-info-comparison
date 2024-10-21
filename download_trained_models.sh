mkdir -p trained_models

for ((n=$1; n<=$2; n++))
do
    echo $n
    wget https://storage.googleapis.com/disentanglement_lib/unsupervised_study_v1/$n.zip 
    unzip $n.zip && rm $n.zip
    mv -f $n/ trained_models/
done
