import pickle
from utils.data_loader import Dataset
from sklearn.model_selection import KFold

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
print(f'train_data is saved at: {all_data_path}')

# with open(train_data_path, 'wb') as f:
#     pickle.dump(train_data, f)
# print(f'train_data is saved at: {train_data_path}')

# with open(val_data_path, 'wb') as f:
#     pickle.dump(val_data, f)
# print(f'val_data is saved at: {val_data_path}')

# with open(test_data_path, 'wb') as f:
#     pickle.dump(test_data, f)
# print(f'test_data is saved at: {test_data_path}')

with open('./data/all_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

kf = KFold(n_splits=2, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(all_data)):
    print(f'\n===== Fold {fold+1} =====')
    train_data = [all_data[i] for i in train_idx]
    test_data = [all_data[i] for i in test_idx]

    train_labeled_data = [i.cid for i in train_data if i.mask == True]
    print('train samples: ', len(train_labeled_data))


    test_labeled_data = [i.cid for i in test_data if i.mask == True]
    print('test samples:', len(test_labeled_data))

    print(len(set(train_labeled_data)))
    print(len(set(test_labeled_data)))
    dup = [i for i in all_data if i.cid == 9816075 or i.cid == 7628]
    for data in dup:
        print(data.x)
        print(data.edge_index)
        print(data.edge_attr)
        break

    # for id in train_labeled_data:
    #     if id in test_labeled_data:
    #         print(id)
    
    # for id in test_labeled_data:
    #     if id in train_data:
    #         print(id)


