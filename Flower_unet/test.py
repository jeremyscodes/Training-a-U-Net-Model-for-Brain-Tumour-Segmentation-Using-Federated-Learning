from sklearn.model_selection import GroupShuffleSplit
list = [[1,2,3],[4,5,6],[7,8],[9,10],[1,2,3],[4,5,6],[7,8],[9,10],[1,2,3],[4,5,6],[7,8],[9,10],[9,10],[9,10],[9,10]]

#list_of_groups = zip(*(iter(list),) * group_size)

 # Group by PatientID
'''grouped = list.groupby('PatientID')
for name, group in grouped:
    print("|"+str(name)+"|")
    print("|"+str(group)+"|")'''

import random

def split_list_of_lists(data, split_percentage):
    """
    Splits a list of lists into two partitions based on a given percentage.
    
    Parameters:
    - data: The list of lists to be split.
    - split_percentage: The percentage of the data to be in the first partition.
    
    Returns:
    - Two lists: The first partition and the second partition.
    """
    # Shuffle the data
    random.shuffle(data)
    
    # Determine the split index
    split_index = int(len(data) * split_percentage)
    
    # Split the data into two partitions
    partition_1 = data[:split_index]
    partition_2 = data[split_index:]
    
    return partition_1, partition_2

p1,p2 = split_list_of_lists(list,0.2)
print("Partition 1")
print(p1)
print("Partition 2")
print(p2)