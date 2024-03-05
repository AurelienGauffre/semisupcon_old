cd ~/semisupcon
. envsemisupcon/bin/activate
git pull
nohup python3 train.py --c ./config/configE1.yaml &
sleep 3
nohup python3 train.py --c ./config/configE2.yaml &
sleep 3
nohup python3 train.py --c ./config/configE2bis.yaml &
wait
