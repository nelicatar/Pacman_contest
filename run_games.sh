i=(0)
while :
do
    python capture.py --delay 0.0 -r baseline_team -b ../../../my_team.py -l RANDOM
    i=$((i+1))
    echo $i
done