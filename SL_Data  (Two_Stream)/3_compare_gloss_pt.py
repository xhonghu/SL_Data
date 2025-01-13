import pickle
import numpy as np
import gzip



def compare_words(str1, str2):
    # Split the two strings into lists of words.
    words1 = str1.split()
    words2 = str2.split()

    # Find the different words.
    differences = []
    for i, (word1, word2) in enumerate(zip(words1, words2)):
        if word1 != word2:
            differences.append((i, word1, word2))

    # Handle the case where one string is longer than the other.
    if len(words1) > len(words2):
        for i in range(len(words2), len(words1)):
            differences.append((i, words1[i], "(no match in str2)"))

    if len(words2) > len(words1):
        for i in range(len(words1), len(words2)):
            differences.append((i, "(no match in str1)", words2[i]))

    return differences


input1 = np.load(f"new_preprocess/phoenix2014-T/train_info.npy", allow_pickle=True).item()
input2 = np.load(f"new_preprocess/phoenix2014-T/dev_info.npy", allow_pickle=True).item()
input3 = np.load(f"new_preprocess/phoenix2014-T/test_info.npy", allow_pickle=True).item()
dict1 = {}
for i in range(len(input1)-1):
    dict1[input1[i]['fileid']]=input1[i]['label']
for i in range(len(input2)-1):
    dict1[input2[i]['fileid']]=input2[i]['label']
for i in range(len(input3)-1):
    dict1[input3[i]['fileid']]=input3[i]['label']

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object
list1 = load_dataset_file('data/phoenix-2014t/phoenix-2014t.train')
list2 = load_dataset_file('data/phoenix-2014t/phoenix-2014t.dev')
list3 = load_dataset_file('data/phoenix-2014t/phoenix-2014t.test')
dict2 = {}
for i in list1:
    dict2[i['name'].replace('train/','')] = i['gloss']
for i in list2:
    dict2[i['name'].replace('dev/','')] = i['gloss']
for i in list3:
    dict2[i['name'].replace('test/','')] = i['gloss']



for k,v in dict1.items():
    if k in dict2:
        a = compare_words(v,dict2[k])
        if len(a)!=0:
            print(k)
            print(v)
            print(dict2[k])
            print(a)
            print()
    else:
        print(k)

###Notice!!!  Two_Stream has two versions, and we are using the one without 'cleaned'. The following can be uncommented for comparison.

# def load_dataset_file(filename):
#     with gzip.open(filename, "rb") as f:
#         loaded_object = pickle.load(f)
#         return loaded_object
# list1 = load_dataset_file('data/phoenix-2014t/phoenix-2014t_cleaned.train')
# list2 = load_dataset_file('data/phoenix-2014t/phoenix-2014t_cleaned.dev')
# list3 = load_dataset_file('data/phoenix-2014t/phoenix-2014t_cleaned.test')
# dict2 = {}
# print(list1[0]['name'].replace('train/',''))
# for i in list1:
#     dict2[i['name'].replace('train/','')] = i['gloss']
# for i in list2:
#     dict2[i['name'].replace('dev/','')] = i['gloss']
# for i in list3:
#     dict2[i['name'].replace('test/','')] = i['gloss']



# for k,v in dict1.items():
#     if k in dict2:
#         a = compare_words(v,dict2[k])
#         if len(a)!=0:
#             print(k)
#             print(v)
#             print(dict2[k])
#             print(a)
#             print()
#     else:
#         print(k)
