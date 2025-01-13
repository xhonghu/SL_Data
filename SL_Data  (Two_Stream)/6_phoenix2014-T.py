from Tokenizer import GlossTokenizer_S2G
import numpy as np
import pickle
import gzip

dict1 = {}
dict1['gloss2id_file'] = 'new_preprocess/phoenix2014-T/gloss2ids.pkl'
gloss_tokenizer = GlossTokenizer_S2G(dict1)
a = gloss_tokenizer.gloss2id.items()
gloss1=[]
for k,v in a:
    gloss1.append(k)
input1 = np.load(f"new_preprocess/phoenix2014-T/train_info.npy", allow_pickle=True).item()
input2 = np.load(f"new_preprocess/phoenix2014-T/dev_info.npy", allow_pickle=True).item()
input3 = np.load(f"new_preprocess/phoenix2014-T/test_info.npy", allow_pickle=True).item()
myset1 = set()
for i in range(len(input1)-1):
    label = input1[i]['label'].split()
    for j in label:
        myset1.add(j.lower())
for i in range(len(input2)-1):
    label = input2[i]['label'].split()
    for j in label:
        myset1.add(j.lower())
for i in range(len(input3)-1):
    label = input3[i]['label'].split()
    for j in label:
        myset1.add(j.lower())
print('The New_preprocss All label Length:',len(list(myset1)))
print('The gloss2is Length:               ',len(gloss1))
print('Label Special Have:                ',myset1-set(gloss1))
print('gloss2is Special Have:             ',set(gloss1)-myset1)
print()


dict2 = {}
dict2['gloss2id_file'] = './data/phoenix-2014t/gloss2ids.pkl'
gloss_tokenizer2 = GlossTokenizer_S2G(dict2)
b = gloss_tokenizer2.gloss2id.items()
gloss2=[]
for k,v in b:
    gloss2.append(k)
def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object
list1 = load_dataset_file('data/phoenix-2014t/phoenix-2014t.train')
list2 = load_dataset_file('data/phoenix-2014t/phoenix-2014t.dev')
list3 = load_dataset_file('data/phoenix-2014t/phoenix-2014t.test')
myset2 =set()
for i in list1:
    label = i['gloss'].split()
    for j in label:
        myset2.add(j.lower())
for i in list2:
    label = i['gloss'].split()
    for j in label:
        myset2.add(j.lower())
for i in list3:
    label = i['gloss'].split()
    for j in label:
        myset2.add(j.lower())
print('The Two_stream(Without Clean version) All label Length:',len(list(myset2)))
print('The gloss2is Length:            ',len(gloss2))
print('Label Special Have:             ',myset2-set(gloss2))
print('gloss2is Special Have:          ',set(gloss2)-myset2)
print()

dict3 = {}
dict3['gloss2id_file'] = './data/phoenix-2014t/gloss2ids_old.pkl'
gloss_tokenizer2 = GlossTokenizer_S2G(dict3)
c = gloss_tokenizer2.gloss2id.items()
gloss3=[]
for k,v in c:
    gloss3.append(k)
def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object
list1 = load_dataset_file('data/phoenix-2014t/phoenix-2014t_cleaned.train')
list2 = load_dataset_file('data/phoenix-2014t/phoenix-2014t_cleaned.dev')
list3 = load_dataset_file('data/phoenix-2014t/phoenix-2014t_cleaned.test')
myset3 =set()
for i in list1:
    label = i['gloss'].split()
    for j in label:
        myset3.add(j.lower())
for i in list2:
    label = i['gloss'].split()
    for j in label:
        myset3.add(j.lower())
for i in list3:
    label = i['gloss'].split()
    for j in label:
        myset3.add(j.lower())
print('The Two_stream(Cleaned version) All label Length:',len(list(myset3)))
print('The gloss2is Length:            ',len(gloss3))
print('Label Special Have:             ',myset3-set(gloss3))
print('gloss2is Special Have:          ',set(gloss3)-myset3)