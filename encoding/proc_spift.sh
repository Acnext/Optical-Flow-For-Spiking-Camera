for ((i=0; i<101; i++))
do
    python3 save_vidar_h5.py -dr /home/huliwen/vidar_data/train/$i
    python3 encoding.py -dr /raid/lwhu/vidarflow/train/$i -sn encoding25 -l 25 -dt 10 &&
    python3 encoding.py -dr /raid/lwhu/vidarflow/train/$i -sn encoding25 -l 25 -dt 20 &&
done
