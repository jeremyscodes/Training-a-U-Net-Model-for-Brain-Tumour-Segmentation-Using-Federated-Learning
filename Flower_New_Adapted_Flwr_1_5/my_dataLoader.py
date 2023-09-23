import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from  keras.utils import Sequence
import tensorflow as tf
#from tqdm import tqdm, trange

#test loader (dummy)

# class SimpleDataGenerator:
#     def __init__(self, batch_size, num_batches):
#         self.batch_size = batch_size
#         self.num_batches = num_batches

#     def generate_data(self):
#         for _ in range(self.num_batches):
#             X = np.random.rand(self.batch_size, 128, 128, 128, 1).astype(np.float32)
#             y = np.random.randint(0, 2, (self.batch_size, 128, 128, 128, 4)).astype(np.float32)
#             yield X, y

    # def get_tf_dataset(self):
    #     return tf.data.Dataset.from_generator(
    #         self.generate_data,
    #         output_signature=(
    #             tf.TensorSpec(shape=(self.batch_size, 128, 128, 128, 1), dtype=tf.float32),
    #             tf.TensorSpec(shape=(self.batch_size, 128, 128, 128, 4), dtype=tf.float32),
    #         )
    #     )


#From 2D unet github dataloader.py
class DatasetGenerator(Sequence):
    """
    TensorFlow Dataset from Python/NumPy Iterator
    """
    
    def __init__(self, filenames, batch_size=8, crop_dim=[240,240], augment=False, seed=816):
        import nibabel as nib
        # Adjust the path for the first filename and load the image
        img = np.array(nib.load(filenames[0]).dataobj)   

        #img = np.array(nib.load(filenames[0]).dataobj) # Load the first image
        self.slice_dim = 2  # We'll assume z-dimension (slice) is last
        # Determine the number of slices (we'll assume this is consistent for the other images)
        self.num_slices_per_scan = img.shape[self.slice_dim]  

        # If crop_dim == -1, then don't crop
        if crop_dim[0] == -1:
            crop_dim[0] = img.shape[0]
        if crop_dim[1] == -1:
            crop_dim[1] = img.shape[1]
        self.crop_dim = crop_dim  

        self.filenames = filenames
        self.batch_size = batch_size

        self.augment = augment
        self.seed = seed
        
        self.num_files = len(self.filenames)
        
        self.ds = self.get_dataset()
    
    def get_tf_dataset(self):
        # Get a batch to inspect its shape
        sample_batch = next(self.ds)
        img_shape = sample_batch[0].shape
        label_shape = sample_batch[1].shape

        # Update the output signature accordingly
        output_signature = (
            tf.TensorSpec(shape=img_shape, dtype=tf.float32),  # img batch
            tf.TensorSpec(shape=label_shape, dtype=tf.float32)  # label batch
        )

        return tf.data.Dataset.from_generator(
            self.generate_batch_from_files,
            output_signature=output_signature
        ).batch(self.batch_size)



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

    def generate_batch_from_files(self):
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

        idx = 0
        idy = 0

        while True:

            """
            Pack N_IMAGES files at a time to queue
            """
            NUM_QUEUED_IMAGES = 1 + self.batch_size // self.num_slices_per_scan  # Get enough for full batch + 1
            
            for idz in range(NUM_QUEUED_IMAGES):

                label_filename = self.filenames[idx]

                #img_filename   = label_filename.replace("_seg.nii.gz", "_flair.nii.gz") # BraTS 2018
                img_filename   = label_filename.replace("labelsTr", "imagesTr")  # Medical Decathlon
                
                img = np.array(nib.load(img_filename).dataobj)
                img = img[:,:,:,0]  # Just take FLAIR channel (channel 0)
                img = self.preprocess_img(img)

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
                
                idx += 1 
                if idx >= len(self.filenames):
                    idx = 0
                    np.random.shuffle(self.filenames) # Shuffle the filenames for the next iteration
            
            img = img_stack
            label = label_stack

            num_slices = img.shape[self.slice_dim]
            
            if self.batch_size > num_slices:
                raise Exception("Batch size {} is greater than"
                                " the number of slices in the image {}."
                                " Data loader cannot be used.".format(self.batch_size, num_slices))

            """
            We can also randomize the slices so that no 2 runs will return the same slice order
            for a given file. This also helps get slices at the end that would be skipped
            if the number of slices is not the same as the batch order.
            """
            if self.augment:
                slice_idx = np.random.choice(range(num_slices), num_slices)
                img = img[:,:,slice_idx]  # Randomize the slices
                label = label[:,:,slice_idx]

            name = self.filenames[idx]
            
            if (idy + self.batch_size) < num_slices:  # We have enough slices for batch
                img_batch, label_batch = img[:,:,idy:idy+self.batch_size], label[:,:,idy:idy+self.batch_size]   

            else:  # We need to pad the batch with slices

                img_batch, label_batch = img[:,:,-self.batch_size:], label[:,:,-self.batch_size:]  # Get remaining slices

            if self.augment:
                img_batch, label_batch = self.augment_data(img_batch, label_batch)
                
            if len(np.shape(img_batch)) == 3:
                img_batch = np.expand_dims(img_batch, axis=-1)
            if len(np.shape(label_batch)) == 3:
                label_batch = np.expand_dims(label_batch, axis=-1)
                
            yield np.transpose(img_batch, [2,0,1,3]).astype(np.float32), np.transpose(label_batch, [2,0,1,3]).astype(np.float32)


            idy += self.batch_size
            if idy >= num_slices: # We finished this file, move to the next
                idy = 0
                idx += 1

            if idx >= len(self.filenames):
                idx = 0
                np.random.shuffle(self.filenames) # Shuffle the filenames for the next iteration

            
                

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
    
    def get_dataset(self):
        """
        Return a dataset
        """
        ds = self.generate_batch_from_files()
        
        return ds  
    
    def __len__(self):
        return (self.num_slices_per_scan * self.num_files)//self.batch_size

    def __getitem__(self, idx):
        return next(self.ds)
        
    def plot_samples(self):
        """
        Plot some random samples
        """
        import matplotlib.pyplot as plt
        
        img, label = next(self.ds)
        
        print(img.shape)
 
        plt.figure(figsize=(10,10))
        
        slice_num = 3
        plt.subplot(2,2,1)
        plt.imshow(img[slice_num,:,:,0]);
        plt.title("MRI, Slice #{}".format(slice_num));

        plt.subplot(2,2,2)
        plt.imshow(label[slice_num,:,:,0]);
        plt.title("Tumor, Slice #{}".format(slice_num));

        slice_num = self.batch_size - 1
        plt.subplot(2,2,3)
        plt.imshow(img[slice_num,:,:,0]);
        plt.title("MRI, Slice #{}".format(slice_num));

        plt.subplot(2,2,4)
        plt.imshow(label[slice_num,:,:,0]);
        plt.title("Tumor, Slice #{}".format(slice_num));

'''
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
    return images, masks'''

'''def get_split_by_id(data):
    'unnecessary since all scans are in a single .nii.gz file for a single brain'
    # Create a GroupShuffleSplit instance
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42) #note 10% testing

    # Get the train and test indices
    train_indices, test_indices = next(gss.split(data, groups=data['PatientID']))

    # Split the train_val data into training and validation sets
    #train_indices, val_indices = next(gss.split(data.iloc[train_val_indices], groups=data.iloc[train_val_indices]['PatientID']))
    # Return the indicies for the data split. Each list contains the indices (rows) for the files in train,val,test
    return train_indices, test_indices'''

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
    print("Start of load_dataset method")
    crop_dim=128  # Original resolution (240)
    batch_size = 128
    seed=816
    train_test_split_val=0.85

    #IID Partitioning

    train_data = pd.read_csv('BRaTSdataset_filenames.csv')
    # Splitting the data into training and testing sets using indices
    train_indices, test_indices = train_test_split(train_data.index, test_size=0.1, random_state=42)

    train_dataset = train_data.iloc[train_indices]
    test_dataset = train_data.iloc[test_indices]

    print("Part A: About to call first split_by_patient_groups")
    client_sets = split_into_n_partitions(train_dataset,num_partitions)
    print("finished splitting into client datasets") 
         #trainsets[0] gives file names for training data for client 0
    '''for t in trainsets:
        print(t)
        print()'''

    # Contains client partitians, inside of which are images_batches and masks_batchs
    trainloaders = []
    valloaders = []

    for client_set in client_sets:
        # Split into training and validation
        train_val_partitions = split_into_train_val(client_set, val_ratio)  
        #print("partitions[0]")
        # Extract file names from each DataFrame in partitions[0] and concatenate them
        # file_names_train = [fname for tup in train_val_partitions[0] for fname in tup[1]['paths'].tolist()]

        file_names_train = train_val_partitions[0]['path'].tolist()
        file_names_train = ["../Task01_BrainTumour" + fname[1:] for fname in file_names_train]


        # file_names_val = [fname for tup in train_val_partitions[1] for fname in tup[1]['paths'].tolist()]
        file_names_val = train_val_partitions[1]['path'].tolist()
        file_names_val = ["../Task01_BrainTumour" + fname[1:] for fname in file_names_val]


        # Initialize DatasetGenerator for training and validation
        
        ds_train_gen = DatasetGenerator(file_names_train, 
                                        batch_size=batch_size,
                                        crop_dim=[crop_dim, crop_dim], 
                                        augment=True, seed=seed)
        # Retrieve the first batch of images and masks
        images_batch, masks_batch = ds_train_gen[0]

        # Access the first image and mask from the batch
        first_image = images_batch[0]
        first_mask = masks_batch[0]

        # Get the shapes of the first image and mask
        first_image_shape = first_image.shape
        first_mask_shape = first_mask.shape

        print("Shape of the first image:", first_image_shape)
        print("Shape of the first mask:", first_mask_shape)


        ds_val_gen = DatasetGenerator(file_names_val, 
                                      batch_size=batch_size,crop_dim=[crop_dim, crop_dim], 
                                      augment=False, seed=seed)

        # Convert to tf.data.Dataset (wrapper for serialization)
        #warning("")
        tf_train_dataset = ds_train_gen.get_tf_dataset()
        tf_val_dataset = ds_val_gen.get_tf_dataset()
        # NOTE : I have added get_tf_Dataset() since the model (manually trained) was able to train. And now in flower it needs to be serializable
        #NOTE : I have removed get_tf_dataset() in order to test if the model loads without the serialized added code
        trainloaders.append(tf_train_dataset)
        valloaders.append(tf_val_dataset)
        

    # Convert test files to actual data
    print("now for test dataloader")
    file_names_test = test_dataset['path'].tolist()
    file_names_test = ["../Task01_BrainTumour" + fname[1:] for fname in file_names_test]

    testloader = DatasetGenerator(file_names_test, 
                           batch_size=batch_size, 
                           crop_dim=[crop_dim, crop_dim], 
                           augment=False, 
                           seed=seed)
    tf_test_dataset = testloader.get_tf_dataset()
    # NOTE : I have added get_tf_Dataset() since the model (manually trained) was able to train. And now in flower it needs to be serializable
    #NOTE : I have removed get_tf_dataset() in order to test if the model loads without the serialized added code
        
    return trainloaders, valloaders, tf_test_dataset
#Finished Fixing on Wednesday 23 August 
# Modified for BRaTS dataset 16 September
def count_images_in_loader(loader, batch_size):
    total_images = 0
    for batch in loader:
        # Using the shape of the batch data to count images
        total_images += batch[0].shape[0]
    return total_images
# def main():
#     trainloaders, valloaders, testloader = load_datasets(num_partitions=2,batch_size=20,  val_ratio=0.1)

#     # Counting images in trainloaders and valloaders for each client
#     # for idx, (trainloader, valloader) in enumerate(zip(trainloaders, valloaders)):
#     #     print(f"Client {idx + 1}:")
#     #     print(f"Number of images in trainloader: {count_images_in_loader(trainloader, 20)}")
#     #     print(f"Number of images in valloader: {count_images_in_loader(valloader, 20)}")
#     #     print("----------------------")

#     # Counting images in testloader
#     print(f"Number of images in testloader: {count_images_in_loader(testloader, 20)}")
# if __name__ == "__main__": main()