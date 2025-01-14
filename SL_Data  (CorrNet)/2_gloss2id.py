import numpy as np
import pickle
import os

gloss2id1 = np.load(f"preprocess/phoenix2014-T/gloss_dict.npy", allow_pickle=True).item()
print(gloss2id1)
index = 0
for k,v in gloss2id1.items():
    gloss2id1[k]  = v[0]
    index = v[0]
gloss2id1['<pad>'] = index + 1
print(gloss2id1)
# 保存到 pkl 文件
if not os.path.exists('new_preprocess/phoenix2014-T/'):
    os.makedirs('new_preprocess/phoenix2014-T/')
with open(f"new_preprocess/phoenix2014-T/gloss2ids.pkl", "wb") as f:  # "wb" 表示以二进制写入模式
    pickle.dump(gloss2id1, f)



gloss2id3 = np.load(f"preprocess/phoenix2014/gloss_dict.npy", allow_pickle=True).item()
print(gloss2id3)
index = 0
for k,v in gloss2id3.items():
    gloss2id3[k]  = v[0]
    index = v[0]
gloss2id3['<pad>'] = index + 1
print(gloss2id3)
if not os.path.exists('new_preprocess/phoenix2014/'):
    os.makedirs('new_preprocess/phoenix2014/')
# 保存到 pkl 文件
with open(f"new_preprocess/phoenix2014/gloss2ids.pkl", "wb") as f:  # "wb" 表示以二进制写入模式
    pickle.dump(gloss2id3, f)