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
    TwoStream_data_dict = {}
    for anno in TwoStream_data:
        key = anno['name'].replace(data_type+'/', "")
        TwoStream_data_dict[key] = anno['text']
    print(data_type)
    print('len(TwoStream_data_dict):  ',len(TwoStream_data_dict))
    print('len(Corrnet_data):  ',len(Corrnet_data))
    
    del Corrnet_data['prefix']
    # Transform Corrnet data with the corresponding gloss labels
    for anno in range(len(Corrnet_data)):
        del Corrnet_data[anno]['original_info']
        Corrnet_data[anno]['text'] = TwoStream_data_dict[Corrnet_data[anno]['fileid']]
    
    print(f"The Example Corrnet {data_type} after transformed annotations:")
    print(Corrnet_data[0])
    print('-------------------------------------------------------------------------------------------------')

    # Default path if not provided
    if output_path is None:
        output_path = f'new_preprocess/phoenix2014-T/{data_type}_info.npy'
    if not os.path.exists('new_preprocess/phoenix2014-T/'):
        os.makedirs('new_preprocess/phoenix2014-T/')

    # Save the transformed annotations to a file
    np.save(output_path, Corrnet_data)

# Corrnet
Corrnet_train = np.load(f"preprocess/phoenix2014-T/train_info.npy", allow_pickle=True).item()
Corrnet_dev = np.load(f"preprocess/phoenix2014-T/dev_info.npy", allow_pickle=True).item()
Corrnet_test = np.load(f"preprocess/phoenix2014-T/test_info.npy", allow_pickle=True).item()
print("The Example Corrnet Original annotations:")
print(Corrnet_train[0])
print()

# TwoStream (Notice!!! We choose the version without 'cleaned'.)
def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object
TwoStream_train = load_dataset_file('data/phoenix-2014t/phoenix-2014t.train')
TwoStream_dev = load_dataset_file('data/phoenix-2014t/phoenix-2014t.dev')
TwoStream_test = load_dataset_file('data/phoenix-2014t/phoenix-2014t.test')
print("The Example TwoStream Original annotations:")
print(TwoStream_train[0])
print('-------------------------------------------------------------------------------------------------')


# Process TrainSet
transform_annotations(TwoStream_train, Corrnet_train, data_type='train')

# Process TestSet
transform_annotations(TwoStream_test, Corrnet_test, data_type='test')

# Process DevSet
transform_annotations(TwoStream_dev, Corrnet_dev, data_type='dev')
