i=(0)
while :
do
    python3 capture.py --delay 0.0 -r baseline_team -b ../../../my_team.py -l RANDOM --quiet
    i=$((i+1))
    echo $i
done