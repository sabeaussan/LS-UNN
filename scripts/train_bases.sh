#!/bin/bash
# train bases Pand/Braccio and UR10/Braccio
cd ../bases
echo $1
if [ -z $1 ];
then 
python3 basesTrainingVAE.py;
python3 basesFittingVAE.py;
else 
python3 basesTrainingVAE.py --run_id $1;
python3 basesFittingVAE.py --run_id $1;  
fi;

