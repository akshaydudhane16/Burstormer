## NTIRE 2021 Burst Super-Resolution Challenge - Track 1 Synthetic
### Training
- Download [Zurich RAW to RGB dataset](http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset).
```
python Burstormer_Track_1_training.py
```
### Developement Phase:
- Download [syn_burst_val](https://data.vision.ee.ethz.ch/bhatg/syn_burst_val.zip) and extract it in root directory.
- Download [Trained model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/akshay_dudhane_mbzuai_ac_ae/ER8mPnjoSIZAnaKA8YyCeE8BA_uQr_73b5qRZx9sh9Rzvw?e=Sc4HFJ) and place it in './Trained_models/Synthetic/Burstormer.ckpt'.
        
```
python Track_1_evaluation.py
```
- Results: stored in './Results/Synthetic/Developement Phase'


## NTIRE 2021 Burst Super-Resolution Challenge - Track 2 Real-world
### Training
- Download [BurstSR train and validation set](https://github.com/goutamgmb/NTIRE21_BURSTSR/blob/master/burstsr_links.md).
- Download [Pretrained Burstormer on synthetic burst SR dataset](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/akshay_dudhane_mbzuai_ac_ae/ER8mPnjoSIZAnaKA8YyCeE8BA_uQr_73b5qRZx9sh9Rzvw?e=Sc4HFJ) and place it in './logs/Track_2/saved_model/Burstormer.ckpt'.
```
python Burstormer_Track_2_training.py
```
### Developement Phase:
- Download [burstsr_dataset](https://data.vision.ee.ethz.ch/bhatg/BurstSRChallenge/val.zip) and extract it in root directory.
- Download [Trained model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/akshay_dudhane_mbzuai_ac_ae/EQK37mvF2axEgMYfJ2-IZUgBHkMxvw1qwqLarlBcRnKLNQ?e=bVAf4F) and place it in './Trained_models/Real/Burstormer.pth'.

```
python Track_2_evaluation.py
```
- Results: stored in './Results/Real/Developement Phase'
