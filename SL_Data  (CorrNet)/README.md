# Data Preparation

1.We will first download the annotation files (data) from [Two_Stream](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork/data).

2.Then, download the annotation files from phoenix2014, phoenix2014-t and phoenix2014-t original dataset..

3.We have already downloaded the files in the project. If you're unsure, you can download them again<br/><br/>


# 1. Generate CorrNet format annotations and add the 'text' field.

``` 
python 1_dataset_preprocess-CSL-Daily.py
```

```  
python 1_dataset_preprocess-T.py
``` 

``` 
python 1_dataset_preprocess.py
```
<br/>

# 2. Convert gloss_dict.npy to gloss2ids.pkl.

``` 
python 1_gloss2id.py
```
<br/>


# 3. Use the script to convert the data into the format we want.
- Modify ['text'] annotations to the original annotation file to prepare for subsequent translation work.
``` 
python 2_csl-daily_preprocess.py
```

- 
```
python 2_phoenix2014_preprocess.py
```
- Modify ['text'] annotations to the original annotation file to prepare for subsequent translation work.
```
python 2_phoenix2014-t_preprocess.py
```
<br/>

# Acknowledgments

Our code is based on [Corrnet](https://github.com/hulianyuyy/CorrNet), [Two_Stream](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork) and [SignGraph](https://github.com/gswycf/SignGraph?tab=readme-ov-file).
