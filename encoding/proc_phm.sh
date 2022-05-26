scene=("ball" "cook" "dice" "dolldrop" "fan" "fly" "hand" "jump" "poker" "top")
for name in ${scene[@]}
do
    python3 save_vidar_h5.py -dr /mnt/huliwen/raid_backup/test/$name &&
    python3 encoding.py -dr /home/huliwen/vidarflow/test/$name -sn encoding25 -l 25 -dt 10 &&
    python3 encoding.py -dr /home/huliwen/vidarflow/test/$name -sn encoding25 -l 25 -dt 20
done
