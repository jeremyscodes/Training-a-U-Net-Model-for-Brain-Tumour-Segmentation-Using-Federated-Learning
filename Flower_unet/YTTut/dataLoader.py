import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from scipy.io import loadmat
import numpy as np
import tensorflow as tf
#from tqdm import tqdm, trange

def load_data(file_names):
    
    # Initialize lists to store data
    images = []
    masks = []

    # Loop over all .mat files
    for fname in file_names:
        # Load the mat file
        mat_data = loadmat('../../Figshare/brainTumorDataPublic_convert_All/'+ fname)#
        image = mat_data['image']
        mask = mat_data['mask']
        # Append data to lists
        images.append(image)
        masks.append(mask)
    print("Finished Loading a scans/masks batch")
    return images, masks

def get_split_by_id(data):
    # Create a GroupShuffleSplit instance
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42) #note 10% testing

    # Get the train and test indices
    train_indices, test_indices = next(gss.split(data, groups=data['PatientID']))

    # Split the train_val data into training and validation sets
    #train_indices, val_indices = next(gss.split(data.iloc[train_val_indices], groups=data.iloc[train_val_indices]['PatientID']))
    # Return the indicies for the data split. Each list contains the indices (rows) for the files in train,val,test
    return train_indices, test_indices

def split_into_train_val_by_patient_groups(data, split_percentage: float):
    """
    Splits data into training and validation sets based on patient groups.
    
    Parameters:
    - data: The DataFrame to be split or a list of (key, group) pairs.
    - split_percentage: The percentage of the groups to be in the validation set.
    
    Returns:
    - Two lists of (key, group) pairs: The training partition and the validation partition.
    """
    
    # Check if data is a DataFrame or list of (key, group) pairs
    if isinstance(data, pd.DataFrame):
        # Group by PatientID
        grouped = data.groupby('PatientID')
        # Convert the grouped object to a list of (key, group) pairs
        patient_groups = list(grouped)
    else:  # Assuming it's a list of (key, group) pairs
        patient_groups = data
    
    # Shuffle the patient groups
    np.random.seed(2023)
    np.random.shuffle(patient_groups)
    
    # Determine the split index
    split_index = max(1, int(len(patient_groups) * split_percentage))
    
    # Split the patient groups into two partitions
    val_set = patient_groups[:split_index]
    train_set = patient_groups[split_index:]
    
    return train_set, val_set



def split_into_n_partitions_by_patient_groups(data, n):
    """
    Splits data into n partitions based on patient groups.
    
    Parameters:
    - data: The DataFrame to be split.
    - n: The number of desired partitions.
    
    Returns:
    - A list of n partitions (each partition is a list of (key, group) pairs).
    """
    # Group by PatientID
    grouped = data.groupby('PatientID')
    
    # Convert the grouped object to a list of (key, group) pairs
    patient_groups = list(grouped)
    
    # Shuffle the patient groups
    np.random.seed(2023)
    np.random.shuffle(patient_groups)
    
    # Calculate size of each partition
    partition_size = len(patient_groups) // n

    # Create n partitions
    partitions = [patient_groups[i * partition_size: (i + 1) * partition_size] for i in range(n)]

    # Handle any remaining groups
    for i, group in enumerate(patient_groups[n * partition_size:]):
        partitions[i].append(group)
    
    return partitions

def load_datasets(num_partitions: int,batch_size: int,  val_ratio: float = 0.1):
    print("Start of load_dataset method")

    #IID Partitioning

    data = pd.read_csv('../../patient_files.csv')
    train_indices, test_indices = get_split_by_id(data)
    # Create lists of file names for training, validation and test sets
    train_set = data.iloc[train_indices]
    test_set = data.iloc[test_indices]['FileName'].tolist()
    #print(test_set)

    #print("Part A: About to call first split_by_patient_groups")
    client_sets = split_into_n_partitions_by_patient_groups(train_set,num_partitions)
    print("finished splitting into client datasets") 
         #trainsets[0] gives file names for training data for client 0
    '''for t in trainsets:
        print(t)
        print()'''

    trainloaders = []
    valloaders = []

    for client_set in client_sets:
        print("new set")
        # Split into training and validation
        train_val_partitions = split_into_train_val_by_patient_groups(client_set, val_ratio)  
        #print("partitions[0]")
        #print(train_val_partitions[0])
        # Extract file names from each DataFrame in partitions[0] and concatenate them
        file_names_train = [fname for tup in train_val_partitions[0] for fname in tup[1]['FileName'].tolist()]
        file_names_val = [fname for tup in train_val_partitions[1] for fname in tup[1]['FileName'].tolist()]
        # Pass the concatenated list to load_data
        x_for_train, y_for_train = load_data(file_names_train)
        x_for_val, y_for_val = load_data(file_names_val)

        # Create TensorFlow datasets
        ds_train = tf.data.Dataset.from_tensor_slices((x_for_train, y_for_train)).shuffle(buffer_size=1000).batch(batch_size)
        ds_val = tf.data.Dataset.from_tensor_slices((x_for_val, y_for_val)).batch(batch_size)
        
        trainloaders.append(ds_train)
        valloaders.append(ds_val)

    # Convert test files to actual data
    print("now for test dataloader")
    
    x_test, y_test = load_data(test_set)
    testloader = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)
    
    return trainloaders, valloaders, testloader
#Finished Fixing on Wednesday 23 August 

