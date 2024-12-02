i=0
while true; do
    echo -n .
    python3 capture.py --delay 0.0 -r ../../../my_minimax_team.py -b ../../../my_team.py -l RANDOM --quiet
    i=$((i+1))
    echo $i
done
