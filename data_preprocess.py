import pickle
from utils.data_loader import Dataset

# paths
labeled_path = './data/labeled_data.csv'
unlabeled_path = './data/unlabeled_data.csv'
searching_space_path = './data/searching_space_data.csv'
train_data_path = './data/train_data.pkl'
val_data_path = './data/val_data.pkl'
test_data_path = './data/test_data.pkl'
all_data_path = './data/all_data.pkl'

# load the dataset
Dataset = Dataset(labeled_path, unlabeled_path, searching_space_path)
all_data = Dataset.load_data()

# save the train, val, test data
with open(all_data_path, 'wb') as f:
    pickle.dump(all_data, f)
print(f'all_data is saved at: {all_data_path}')

# with open(train_data_path, 'wb') as f:
#     pickle.dump(train_data, f)
# print(f'train_data is saved at: {train_data_path}')

# with open(val_data_path, 'wb') as f:
#     pickle.dump(val_data, f)
# print(f'val_data is saved at: {val_data_path}')

# with open(test_data_path, 'wb') as f:
#     pickle.dump(test_data, f)
# print(f'test_data is saved at: {test_data_path}')


