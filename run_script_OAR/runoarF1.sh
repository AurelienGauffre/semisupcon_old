cd ~/semisupcon
. envsemisupcon/bin/activate
git pull
DSDIR="/home/aptikal/gauffrea/datasets"
DSDIR_CUSTOM="/home/aptikal/gauffrea/datasets"
export DSDIR
export DSDIR_CUSTOM
python3 train.py --c ./config/configF1.yaml
