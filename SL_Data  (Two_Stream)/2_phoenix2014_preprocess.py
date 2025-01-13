import shutil
import numpy as np
import pickle
import gzip
import os

def transform_annotations(TwoStream_data, Corrnet_data, data_type='train', output_path=None):
    """
    Transform annotations for the Corrnet dataset by mapping gloss labels from TwoStream data.

    Args:
    - TwoStream_data (list): List of annotations for the TwoStream data (train, test, or dev).
    - Corrnet_data (list): List of annotations for the Corrnet data (train, test, or dev).
    - data_type (str): Specifies whether the data is 'train', 'test', or 'dev'. Default is 'train'.
    - output_path (str): Path to save the processed annotations after transformation. If not specified, default will be used.

    Returns:
    - None
    """

    # Create a dictionary to map folder names to gloss values
    # TwoStream Special have: 13April_2011_Wednesday_tagesschau_default-14
    TwoStream_data_dict = {}
    Special_TwoStream = {}
    for anno in TwoStream_data:
        key = anno['folder'].replace("/1/*.png", "")
        TwoStream_data_dict[key] = anno['gloss']
        if key == '13April_2011_Wednesday_tagesschau_default-14':
            Special_TwoStream = anno
    print(data_type)
    print('len(TwoStream_data_dict):  ',len(TwoStream_data_dict))
    print('len(Corrnet_data):  ',len(Corrnet_data))
    print()

    # Transform Corrnet data with the corresponding gloss labels
    del Corrnet_data['prefix']
    for anno in range(len(Corrnet_data)):
        Corrnet_data[anno]['label'] = TwoStream_data_dict[Corrnet_data[anno]['fileid']]
        del TwoStream_data_dict[Corrnet_data[anno]['fileid']]
    
    if len(TwoStream_data_dict)!=0:
        print('TwoStream Special have:',Special_TwoStream)
        add_dict = {}
        add_dict['fileid']=Special_TwoStream['folder'].replace("/1/*.png","")
        add_dict['folder']='train/' + Special_TwoStream['folder']
        add_dict['signer']=Special_TwoStream['signer']
        add_dict['label']=Special_TwoStream['gloss']
        add_dict['num_frames']=Special_TwoStream['num_frames']
        Corrnet_data[5671] = add_dict
        print('\nWhich One Add Two_Stream Special to CorrNet:',Corrnet_data[5671])
        print()
    print(f"The Example Corrnet {data_type} after transformed annotations:")
    print(Corrnet_data[0])
    print('-------------------------------------------------------------------------------------------------')

    # Default path if not provided
    if output_path is None:
        output_path = f'new_preprocess/phoenix2014/{data_type}_info.npy'
    if not os.path.exists('new_preprocess/phoenix2014/'):
        os.makedirs('new_preprocess/phoenix2014/')

    # Save the transformed annotations to a file
    np.save(output_path, Corrnet_data)

# Corrnet
Corrnet_train = np.load(f"preprocess/phoenix2014/train_info.npy", allow_pickle=True).item()
Corrnet_dev = np.load(f"preprocess/phoenix2014/dev_info.npy", allow_pickle=True).item()
Corrnet_test = np.load(f"preprocess/phoenix2014/test_info.npy", allow_pickle=True).item()
print("The Example Corrnet Original annotations:")
print(Corrnet_train[0])
print()

# TwoStream
def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object
TwoStream_train = load_dataset_file('data/phoenix-2014/phoenix-2014.train')
TwoStream_dev = load_dataset_file('data/phoenix-2014/phoenix-2014.dev')
TwoStream_test = load_dataset_file('data/phoenix-2014/phoenix-2014.test')
print("The Example TwoStream Original annotations:")
print(TwoStream_train[0])
print('-------------------------------------------------------------------------------------------------')

# Process TrainSet
transform_annotations(TwoStream_train, Corrnet_train, data_type='train')

# Process TestSet
transform_annotations(TwoStream_test, Corrnet_test, data_type='test')

# Process DevSet
transform_annotations(TwoStream_dev, Corrnet_dev, data_type='dev')

shutil.copy2('data/phoenix-2014/gloss2ids.pkl', 'new_preprocess/phoenix2014/')
