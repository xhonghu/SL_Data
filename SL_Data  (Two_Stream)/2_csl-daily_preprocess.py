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
    TwoStream_data_dict = {}
    Special_dict = {}
    # Special_dict For Corrnet dataset have not S000005_P0004_T00, but S00007_P0003_T00
    # TwoStream dataset have not S00007_P0003_T00, but S000005_P0004_T00
    for anno in TwoStream_data:
        key = anno['name']
        if key == 'S000005_P0004_T00':
            Special_dict[key] = [anno['name'],anno['gloss'],anno['text'],anno['num_frames'],anno['signer']]
        TwoStream_data_dict[key] = [anno['gloss'],anno['text'],anno['signer']]

    print(data_type)
    print('len(TwoStream_data_dict):  ',len(TwoStream_data_dict))
    print('len(Corrnet_data):  ',len(Corrnet_data))

    # Transform Corrnet data with the corresponding gloss labels
    ### Notice!!! Corrnet dataset have not S000005_P0004_T00, but S00007_P0003_T00
    ### Notice!!! TwoStream dataset have not S00007_P0003_T00, but S000005_P0004_T00
    for anno in range(len(Corrnet_data)):
        if Corrnet_data[anno]['fileid'] in TwoStream_data_dict:
            Corrnet_data[anno]['label'] = TwoStream_data_dict[Corrnet_data[anno]['fileid']][0]
            Corrnet_data[anno]['text'] = TwoStream_data_dict[Corrnet_data[anno]['fileid']][1]
            Corrnet_data[anno]['signer'] = TwoStream_data_dict[Corrnet_data[anno]['fileid']][2]
            del TwoStream_data_dict[Corrnet_data[anno]['fileid']]
        else:
            print("\nCorrNet Special have  :",Corrnet_data[anno])

    print('\nTwoStream Special have:',TwoStream_data_dict)
    print(f"\nThe Example Corrnet {data_type} after transformed annotations:")
    print(Corrnet_data[0])
    print('-------------------------------------------------------------------------------------------------')

    # Default path if not provided
    if output_path is None:
        output_path = f'new_preprocess/CSL-Daily/{data_type}_info.npy'
    if not os.path.exists('new_preprocess/CSL-Daily/'):
        os.makedirs('new_preprocess/CSL-Daily/')        

    # Save the transformed annotations to a file
    np.save(output_path, Corrnet_data)

# Corrnet
Corrnet_train = np.load(f"preprocess/CSL-Daily/train_info.npy", allow_pickle=True).item()
Corrnet_dev = np.load(f"preprocess/CSL-Daily/dev_info.npy", allow_pickle=True).item()
Corrnet_test = np.load(f"preprocess/CSL-Daily/test_info.npy", allow_pickle=True).item()
print("The Example Corrnet Original annotations:")
print(Corrnet_train[0])
print()
# TwoStream
def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object
TwoStream_train = load_dataset_file('data/csl-daily/csl-daily.train')
TwoStream_dev = load_dataset_file('data/csl-daily/csl-daily.dev')
TwoStream_test = load_dataset_file('data/csl-daily/csl-daily.test')
print("The Example TwoStream Original annotations:")
print(TwoStream_train[0])
print('-------------------------------------------------------------------------------------------------')


# Process TrainSet
transform_annotations(TwoStream_train, Corrnet_train, data_type='train')

# Process TestSet
transform_annotations(TwoStream_test, Corrnet_test, data_type='test')

# Process DevSet
transform_annotations(TwoStream_dev, Corrnet_dev, data_type='dev')

# Copy the gloss2id from Two_Stream.
shutil.copy2('data/csl-daily/gloss2ids.pkl', 'new_preprocess/CSL-Daily/')
