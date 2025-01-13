from Tokenizer import GlossTokenizer_S2G

#If the lengths are 1235, 1124, and 2004 respectively, then there is no issue, and the gloss_tokenizer is consistent with Two_Stream.
dict1 = {}
dict1['gloss2id_file'] = 'new_preprocess/phoenix2014/gloss2ids.pkl'
gloss_tokenizer = GlossTokenizer_S2G(dict1)
print('phoenix2014     len(gloss_tokenizer):  ',len(gloss_tokenizer))

dict2 = {}
dict2['gloss2id_file'] = 'new_preprocess/phoenix2014-T/gloss2ids.pkl'
gloss_tokenizer2 = GlossTokenizer_S2G(dict2)
print('phoenix2014-T   len(gloss_tokenizer):  ',len(gloss_tokenizer2))


dict3 = {}
dict3['gloss2id_file'] = 'new_preprocess/CSL-Daily/gloss2ids.pkl'
gloss_tokenizer3 = GlossTokenizer_S2G(dict3)
print('csl-daily       len(gloss_tokenizer):  ',len(gloss_tokenizer3))