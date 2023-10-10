import tensorflow as tf
import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.model_selection import train_test_split
'''State Storage: The generator does not store the actual image and label data in its state.
 It only stores the filenames, batch size, and other configuration parameters needed to generate the batches.'''
class SerializableDatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, filenames, batch_size=8, crop_dim=[240, 240], augment=False, seed=816):
        self.filenames = filenames
        self.batch_size = batch_size
        self.crop_dim = crop_dim
        self.augment = augment
        self.seed = seed
        self.indexes = np.arange(len(self.filenames))
        self.slice_dim = 2
        img = np.array(nib.load(filenames[0]).dataobj)  
        self.num_slices_per_scan = img.shape[self.slice_dim]
        self.num_files = len(self.filenames)
        # print(f"Number of 3D volumes: {len(filenames)}")    


   
        
    def __len__(self):
        return int(np.ceil(len(self.filenames) / self.batch_size))


    def __getitem__(self, index):
  
        self.idy = 0

        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_filenames = self.filenames[start:end]
        
        batch_data, batch_labels = self.generate_batch_from_files(batch_filenames)
        """print("batch_data shape",batch_data.shape)
        print("batch_labels shape",batch_labels.shape)"""
        #batch_data=np.array(batch_data)
        #batch_labels=np.array(batch_labels)
        self.idy += self.batch_size
        if self.idy >= self.num_slices_per_scan:
            self.idy = 0
        return batch_data, batch_labels

    def on_epoch_end(self):
        np.random.seed(self.seed)
        indexes_copy = np.copy(self.indexes)
        np.random.shuffle(indexes_copy)
        self.indexes = indexes_copy        
    # def on_epoch_end(self):
    #     np.random.seed(self.seed)
    #     np.random.shuffle(self.indexes)
    
    def preprocess_img(self, img):
        """
        Preprocessing for the image
        z-score normalize
        """
        return (img - img.mean()) / img.std()

    def preprocess_label(self, label):
        """
        Predict whole tumor. If you want to predict tumor sections, then 
        just comment this out.
        """
        label[label > 0] = 1.0

        return label
    
    def augment_data(self, img, msk):
        """
        Data augmentation
        Flip image and mask. Rotate image and mask.
        """
        
        if np.random.rand() > 0.5:
            ax = np.random.choice([0,1])
            img = np.flip(img, ax)
            msk = np.flip(msk, ax)

        if np.random.rand() > 0.5:
            rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees

            img = np.rot90(img, rot, axes=[0,1])  # Rotate axes 0 and 1
            msk = np.rot90(msk, rot, axes=[0,1])  # Rotate axes 0 and 1

        return img, msk

    def crop_input(self, img, msk):
            """
            Randomly crop the image and mask
            """

            slices = []

            # Do we randomize?
            is_random = self.augment and np.random.rand() > 0.5

            for idx, idy in enumerate(range(2)):  # Go through each dimension

                cropLen = self.crop_dim[idx]
                imgLen = img.shape[idy]

                start = (imgLen-cropLen)//2

                ratio_crop = 0.20  # Crop up this this % of pixels for offset
                # Number of pixels to offset crop in this dimension
                offset = int(np.floor(start*ratio_crop))

                if offset > 0:
                    if is_random:
                        start += np.random.choice(range(-offset, offset))
                        if ((start + cropLen) > imgLen):  # Don't fall off the image
                            start = (imgLen-cropLen)//2
                else:
                    start = 0

                slices.append(slice(start, start+cropLen))

            return img[tuple(slices)], msk[tuple(slices)]

        
    def generate_batch_from_files(self,batch_filenames):
        """
        Python generator which goes through a list of filenames to load.
        The files are 3D image (slice is dimension index 2 by default). However,
        we need to yield them as a batch of 2D slices. This generator
        keeps yielding a batch of 2D slices at a time until the 3D image is 
        complete and then moves to the next 3D image in the filenames.
        An optional `randomize_slices` allows the user to randomize the 3D image 
        slices after loading if desired.
        """
        import nibabel as nib

        np.random.seed(self.seed)  # Set a random seed

  
        
        # print(f"Number of 3D volumes: {len(batch_filenames)}")

        for idz in range(0, len(batch_filenames)):

            img_filename = batch_filenames[idz]

            img = np.array(nib.load(img_filename).dataobj)
            img = img[:,:,:,0]  # Just take FLAIR channel (channel 0)
            img = self.preprocess_img(img)
            label_filename = img_filename.replace("imagesTr", "labelsTr")
            label = np.array(nib.load(label_filename).dataobj)
            label = self.preprocess_label(label)
            
            # Crop input and label
            img, label = self.crop_input(img, label)
            

            if idz == 0:
                img_stack = img
                label_stack = label

            else:

                img_stack = np.concatenate((img_stack,img), axis=self.slice_dim)
                label_stack = np.concatenate((label_stack,label), axis=self.slice_dim)
        
        # print(f"img_stack shape: {img_stack.shape}")
        # print(f"label_stack shape: {label_stack.shape}")


        img = img_stack
        label = label_stack

        num_slices = img.shape[self.slice_dim]
        self.num_slices = num_slices
        
        if self.batch_size > num_slices:
            raise Exception("Batch size {} is greater than"
                            " the number of slices in the image {}."
                            " Data loader cannot be used.".format(self.batch_size, num_slices))

        """
        We can also ranrandomize the slices so that no 2 runs will return the same slice order
        for a given file. This also helps get slices at the end that would be skipped
        if the number of slices is not the same as the batch order.
        """
        if self.augment:
            slice_idx = np.random.choice(range(num_slices), num_slices)
            img = img[:,:,slice_idx]  # Randomize the slices
            label = label[:,:,slice_idx]

        if (self.idy + self.batch_size) < num_slices:  # We have enough slices for a batch
                img_batch, label_batch = img[:, :, self.idy:self.idy + self.batch_size], label[:, :, self.idy:self.idy + self.batch_size]
        else:  # We need to pad the batch with slices

            img_batch, label_batch = img[:,:,-self.batch_size:], label[:,:,-self.batch_size:]  # Get remaining slices

        if self.augment:
            img_batch, label_batch = self.augment_data(img_batch, label_batch)
    
        if len(np.shape(img_batch)) == 3:
            img_batch = np.expand_dims(img_batch, axis=-1)
        if len(np.shape(label_batch)) == 3:
            label_batch = np.expand_dims(label_batch, axis=-1)
        #print("here")
        return np.transpose(img_batch, [2,0,1,3]).astype(np.float32), np.transpose(label_batch, [2,0,1,3]).astype(np.float32)

        'returns NumPy arrays so no conversion needed!'
    def get_input_shape(self):
        """
        Get image shape
        """
        return [self.crop_dim[0], self.crop_dim[1], 1]
        
    def get_output_shape(self):
        """
        Get label shape
        """
        return [self.crop_dim[0], self.crop_dim[1], 1] 
    


def split_into_train_val(data, split_percentage: float):
    """
    Splits data into training and validation sets.
    
    Parameters:
    - data: The DataFrame to be split.
    - split_percentage: The percentage of the data to be in the validation set.
    
    Returns:
    - Two DataFrames: The training set and the validation set.
    """
    
    # Shuffle the data
    shuffled_data = data.sample(frac=1, random_state=2023).reset_index(drop=True)
    
    # Determine the split index
    split_index = max(1, int(len(shuffled_data) * split_percentage))
    
    # Split the data into training and validation sets
    val_set = shuffled_data[:split_index]
    train_set = shuffled_data[split_index:]
    
    return train_set, val_set



def split_into_n_partitions(data, n):
    """
    Splits data into n partitions.
    
    Parameters:
    - data: The DataFrame to be split.
    - n: The number of desired partitions.
    
    Returns:
    - A list of n DataFrame partitions.
    """
    # Shuffle the data
    shuffled_data = data.sample(frac=1, random_state=2023).reset_index(drop=True)
    
    # Calculate size of each partition
    partition_size = len(shuffled_data) // n

    # Create n partitions
    partitions = [shuffled_data.iloc[i * partition_size: (i + 1) * partition_size] for i in range(n)]

    # Handle any remaining rows
    for i, row in enumerate(shuffled_data.iloc[n * partition_size:].iterrows()):
        partitions[i] = pd.concat([partitions[i], pd.DataFrame([row[1]])], ignore_index=True)

    return partitions


def load_datasets(num_partitions: int,batch_size: int,  val_ratio: float = 0.1):
    crop_dim=128  # Original resolution (240)
    seed=816
    train_test_split_val=0.85

    #IID Partitioning

    train_data = pd.read_csv('BRaTSdataset_filenames.csv')
    # Splitting the data into training and testing sets using indices
    train_indices, test_indices = train_test_split(train_data.index, test_size=0.2, random_state=42)
    #split train_indices into train and validation
    train_indices, val_indices = train_test_split(train_indices, test_size=0.25, random_state=42)
    # Train: 60%, Val: 20%, Test: 20%

    train_dataset = train_data.iloc[train_indices]
    val_indices = train_data.iloc[val_indices]
    test_dataset = train_data.iloc[test_indices]

    client_sets = split_into_n_partitions(train_dataset,num_partitions)
         #trainsets[0] gives file names for training data for client 0
    '''for t in trainsets:
        print(t)
        print()'''

    # Contains client partitians, inside of which are images_batches and masks_batchs
    trainloaders = []

    #valloaders = []
    first = True
    for client_set in client_sets:
        # Split into training and validation

        # train_val_partitions = split_into_train_val(client_set, val_ratio)  
        #print("partitions[0]")
        # Extract file names from each DataFrame in partitions[0] and concatenate them
        # file_names_train = [fname for tup in train_val_partitions[0] for fname in tup[1]['paths'].tolist()]

        file_names_train = client_set[0]['path'].tolist()
        file_names_train = ["../Task01_BrainTumour" + fname[1:] for fname in file_names_train]


        # file_names_val = [fname for tup in train_val_partitions[1] for fname in tup[1]['paths'].tolist()]
        # file_names_val = client_set[1]['path'].tolist()
        # file_names_val = ["../Task01_BrainTumour" + fname[1:] for fname in file_names_val]


        # Initialize DatasetGenerator for training and validation
        
        ds_train_gen = SerializableDatasetGenerator(file_names_train, 
                                        batch_size=batch_size,
                                        crop_dim=[crop_dim, crop_dim], 
                                        augment=True, seed=seed)
        
        
        if first:
            input_shape, output_shape = ds_train_gen.get_input_shape(), ds_train_gen.get_output_shape()
            first=False
        
        
        trainloaders.append(ds_train_gen)
        # valloaders.append(ds_val_gen)
        
    # TODO Validation
    file_names_val = val_indices['path'].tolist()
    file_names_val = ["../Task01_BrainTumour" + fname[1:] for fname in file_names_val]

    valloader = SerializableDatasetGenerator(file_names_val, 
                                batch_size=batch_size,
                                crop_dim=[crop_dim, crop_dim], 
                                augment=False, 
                                seed=seed)

     # Convert test files to actual data
    file_names_test = test_dataset['path'].tolist()
    file_names_test = ["../Task01_BrainTumour" + fname[1:] for fname in file_names_test]

    testloader = SerializableDatasetGenerator(file_names_test, 
                           batch_size=batch_size, 
                           crop_dim=[crop_dim, crop_dim], 
                           augment=False, 
                           seed=seed)
        
    return trainloaders, valloader, testloader, input_shape, output_shape
#Finished Fixing on Wednesday 23 August 
# Modified for BRaTS dataset 16 September
def count_images_in_loader(loader):
    total_images = 0
    num_batches = len(loader)
    if num_batches > 0:
        # Assuming the number of slices per scan is consistent across all images
        # and is known beforehand. Replace with the actual number of slices per scan.
        num_slices_per_scan = loader.num_slices_per_scan
        
        # Calculate the total number of images (slices) in the loader for all batches except the last one
        total_images = (num_batches - 1) * loader.batch_size * num_slices_per_scan
        
        # Handle the last batch separately as it may not be a full batch
        # Calculate the number of images in the last batch without loading it
        num_images_in_last_batch = (len(loader.filenames) % loader.batch_size) * num_slices_per_scan
        total_images += num_images_in_last_batch
        
    return total_images

def count_images_in_loader2(loader):
    total_images = 0
    num_batches = len(loader)
    num_slices_per_scan = loader.num_slices_per_scan  # 155 as verified
    
    for idx in range(num_batches):
        start_idx = idx * loader.batch_size
        end_idx = (idx + 1) * loader.batch_size
        actual_batch_size = min(end_idx, len(loader.filenames)) - start_idx
        total_images += actual_batch_size * num_slices_per_scan  # total_images in this batch
    
    return total_images



def main():
    
    
    trainloaders, valloaders, testloader, input_shape, output_shape = load_datasets(num_partitions=8,batch_size=20,  val_ratio=0.1)
    print("Loaded")
   
    # Counting images in trainloaders and valloaders for each client
    for idx, (trainloader, valloader) in enumerate(zip(trainloaders, valloaders)):
        print(f"Client {idx + 1}:")
        print(f"Number of images in trainloader: {count_images_in_loader2(trainloader)}")
        print(f"Number of images in valloader: {count_images_in_loader2(valloader)}")
        print("----------------------")

    # # Counting images in testloader
    print(f"Number of images in testloader: {count_images_in_loader2(testloader)}")
if __name__ == "__main__": main()