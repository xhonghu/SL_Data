import pickle
with open('annotations/csl-daily/csl2020ct_v2.pkl', 'rb') as f:
    data = pickle.load(f)
print(data.keys())
print(data['info'][0])
print(data['info'][0]['name'])
for i in data['info']:
    if i['name']=='S007268_P0007_T00':
        print(i)