# Data Preparation

1.We will first download the annotation files (data) from [Two_Stream](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork/data).

2.Then, We will first download the annotation files (preprocess) from [SignGraph](https://github.com/gswycf/SignGraph/tree/main/preprocess).

3.We have already downloaded the files in the project. If you're unsure, you can download them again<br/><br/>

# 1. Convert gloss_dict.npy to gloss2ids.pkl.

``` 
python 1_gloss2id.py
```
<br/>


# 2. Use the script to convert the data into the format we want.
- Add ['text'] annotations to the original annotation file to prepare for subsequent translation work.
- Remove redundant information 'original_info'.
``` 
python 2_csl-daily_preprocess.py
```

- Remove redundant information 'original_info','prefix'.
```
python 2_phoenix2014_preprocess.py
```
- Add ['text'] annotations to the original annotation file to prepare for subsequent translation work.
- Remove redundant information 'original_info','prefix'.
```
python 2_phoenix2014-t_preprocess.py
```
<br/>

# Acknowledgments

Our code is based on [Corrnet](https://github.com/hulianyuyy/CorrNet), [Two_Stream](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork) and [SignGraph](https://github.com/gswycf/SignGraph?tab=readme-ov-file).
