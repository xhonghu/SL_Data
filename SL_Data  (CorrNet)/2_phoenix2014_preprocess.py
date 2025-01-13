import numpy as np
import os

def transform_annotations( Corrnet_data, data_type='train', output_path=None):
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

    print(data_type)
    print('len(Corrnet_data):  ',len(Corrnet_data))
    print()
    del Corrnet_data['prefix']
    for anno in range(len(Corrnet_data)):
        del Corrnet_data[anno]['original_info']
    print(Corrnet_data[0])
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

# Process TrainSet
transform_annotations(Corrnet_train, data_type='train')

# Process TestSet
transform_annotations(Corrnet_test, data_type='test')

# Process DevSet
transform_annotations(Corrnet_dev, data_type='dev')
