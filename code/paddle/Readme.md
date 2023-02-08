## Pipeline

# numericNLG

```powershell
cd code/paddle
python preprocess.py numeircNLG
sh pipeline.sh 21 3 2 128 small 1e-5 -1 4 0,1,2,3 4 numericNLG first
```

# Totto

```powershell
cd code/pytorch
python preprocess.py Totto
sh pipeline.sh 21 3 2 128 small 1e-5 -1 4 0,1,2,3 4 Totto first
```
