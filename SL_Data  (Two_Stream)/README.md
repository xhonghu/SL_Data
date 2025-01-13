# Data Preparation

1.We will first download the annotation files from [Two_Stream](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork/data).

2.Then, download the annotation files from phoenix2014, phoenix2014-t and phoenix2014-t original dataset..

3.We have already downloaded the files in the project. If you're unsure, you can download them again<br/><br/>

# 1. Generate annotations in the CorrNet framework format based on the original annotation files.

The processing script is from [CorrNet](https://github.com/hulianyuyy/CorrNet/tree/main/preprocess), with slight modifications: we removed the image processing module and added text annotations.
``` 
python 1_dataset_preprocess-CSL-Daily.py

python 1_dataset_preprocess-T.py

python 1_dataset_preprocess.py
```
<br/>


# 2. Use the script to convert the data into the format we want.
``` 
python 2_csl-daily_preprocess.py

python 2_phoenix2014_preprocess.py

python 2_phoenix2014-t_preprocess.py
```
<br/>

# 3. Verify if there is any difference between the converted gloss and the one in Two_Stream.

If there is no output, then the conversion result is likely fine.
``` 
python 3_compare_gloss_csl.py

python 3_compare_gloss_p.py

python 3_compare_gloss_pt.py
```
Attention!!! In Two_Stream, there are both Cleaned and non-Cleaned versions of the annotations. We are using the non-Cleaned version. You can uncomment the comparison for the Cleaned version to see the differences between the two.<br/><br/>

# 4.Verify if there is any difference between the converted text and the one in Two_Stream.

If there is no output, then the conversion result is likely fine.
``` 
python 4_compare_text_csl.py

python 4_compare_text_pt.py
```
Attention!!! In Two_Stream, there are both Cleaned and non-Cleaned versions of the annotations. We are using the non-Cleaned version. You can uncomment the comparison for the Cleaned version to see the differences between the two.<br/><br/>
# 5. Check if there are any issues with the processing of gloss2id.
``` 
python 5_test_id2gloss.py
```
If the lengths are 1235, 1124, and 2004 respectively, then there is no issue, and the gloss_tokenizer is consistent with Two_Stream.<br/><br/>

# 6. Verify if gloss2id corresponds to the annotations.
``` 
python 6_csl-daily.py

python 6_phoenix2014-T.py

python 6_phoenix2014.py
```
<br/>

# Acknowledgments

Our code is based on [Corrnet](https://github.com/hulianyuyy/CorrNet) and [Two_Stream](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork).
