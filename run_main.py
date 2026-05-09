import os
import torch
import argparse
import warnings
import pandas as pd
import random
import numpy as np

from rdkit import Chem
from torch_geometric.loader import DataLoader
from utils.data_loader import load_data
from utils.recommends import Recommender, do_recommendation
from utils.post_screening import PostScreening
from utils.USL import USL
from utils.PCL import PCL
from rdkit.Chem import DataStructs
from utils.tools import mol_to_fp

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Train the model")
# basic configs
parser.add_argument('--method', type=str, default='full_model', help='recommendation method')
parser.add_argument('--data_path', type=str, default='./data/all_data.pkl', help='Path to the preprocessed data')
parser.add_argument('--save_path', type=str, default='checkpoints_origin_backup', help='')
parser.add_argument('--device', type=str, default='cuda:7' if torch.cuda.is_available() else 'cpu', help='Device to use for training')


# baseline configs
parser.add_argument('--num_select', type=int, default=30800 , help='Number of molecules to select')


# unsupervised learning configs
parser.add_argument('--analysis', type=bool, default=False, help='Wether to print the summary of the dataset')
parser.add_argument('--usl_batch_size', type=int, default=256, help='Batch size for training')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
parser.add_argument('--usl_learning_rate', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--usl_hidden_channels', type=int, default=256, help='Number of hidden channels')
parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs')
parser.add_argument('--dropout', type=float, default=0.5, help='Value of dropout')
parser.add_argument('--training_types', type=str, default='Unsupervised learning', help='training_types')
parser.add_argument('--models', type=str, default='GINE', help='Training models')
parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension of attention')
parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for attention')
parser.add_argument('--desp_dim', type=int, default=217, help='Number of descriptors')
parser.add_argument('--retrain_usl', type=bool, default=False, help='retrain the usl models')
parser.add_argument('--usl_trials', type=int, default=10, help='Number of trials for unsupervised learning')


# graph augmentation configs
parser.add_argument('--aug_types', type=str, default='all', help='augmentation types')
parser.add_argument('--shuffle_ratio', type=float, default=0.2, help='shuffle ratio')
parser.add_argument('--node_drop_ratio', type=float, default=0.2, help='node drop ratio')
parser.add_argument('--noise_ratio', type=float, default=0.2, help='noise_ratio')
parser.add_argument('--noise_std', type=float, default=0.1, help='noise_std')
parser.add_argument('--edge_drop_ratio', type=float, default=0.1, help='edge drop ratio')
parser.add_argument('--edge_add_ratio', type=float, default=0.05, help='edge add ratio')
parser.add_argument('--alpha', type=float, default=0.15, help='PPR alpha')
parser.add_argument('--PPR_drop_ratio', type=float, default=0.2, help='PPR_drop_ratio')
parser.add_argument('--PPR_add_ratio', type=float, default=0.2, help='PPR_add_ratio')
parser.add_argument('--K', type=int, default=10, help='PPR K')
parser.add_argument('--random_state', type=int, default=42, help='data split random seed')
parser.add_argument('--test_size', type=float, default=0.2, help='test set size')


# prototypes contrastive learning configs
parser.add_argument('--max_cluster', type=int, default=10, help='max cluster number')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature coefficient for prototypes')
parser.add_argument('--proto_epoch', type=int, default=300, help='Number of training epochs')
parser.add_argument('--r', type=int, default=10000, help='number of randomly select neg prototypes')
parser.add_argument('--proto_training_types', type=str, default='Prototype contrastive learning', help='training_types')
parser.add_argument('--proto_models', type=str, default='GINE', help='model name for PCL')
parser.add_argument('--pcl_hidden_channels', type=int, default=256, help='Number of hidden channels')
parser.add_argument('--pcl_learning_rate', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--pcl_batch_size', type=int, default=1024, help='Batch size for training')
parser.add_argument('--threshold', type=float, default=0.3, help='threshold')
parser.add_argument('--topk', type=int, default=25, help='top k samples for each prototype')
parser.add_argument('--pcl_trials', type=int, default=10, help='Number of trials for unsupervised learning')
parser.add_argument('--save_proto_drift', type=bool, default=False, help='Whether to save prototype drift information')
parser.add_argument('--EMA', type=bool, default=True, help='Whether to activate Exponential Moving Average (EMA) for prototype updating')
parser.add_argument('--use_decor_loss', type=bool, default=True, help='Whether to activate decorrelation loss for prototype learning')
parser.add_argument('--use_topk', type=bool, default=True, help='Whether to activate top-k selection for prototype learning')



# post-screening configs
parser.add_argument('--save_molecules', type=bool, default=True, help='Whether to save the recommended molecules after post-screening')
parser.add_argument('--additive_json_path', type=str, default='./V3/processed_data/additives.json', help='Path to the JSON file containing additive data')
parser.add_argument('--best_prototype_path', type=str, default='./result_files/proto_table_trial_7.csv', help='Path to the CSV file containing the best prototypes')
parser.add_argument('--searching_space_path', type=str, default='./data/searching_space_data_V2.csv', help='Path to the CSV file containing the searching space data')
parser.add_argument('--post-screening_output_path', type=str, default='./outputs/', help='Path to the post-screening output file')



args = parser.parse_args()


def random_recommendation(unlabeled_samples, random_state):
    # random recommend samples
    random.seed(random_state)
    recommended_samples = random.sample(unlabeled_samples, args.num_select)

    random_loader = DataLoader(recommended_samples, batch_size=args.pcl_batch_size, shuffle=False)

    return random_loader


def morgan_recommendation(positive_samples, unlabeled_samples):
    print(positive_samples[:5])


    unl_fps = [mol_to_fp(Chem.MolFromSmiles(s)) for s in unlabeled_samples]
    pos_fps = [mol_to_fp(Chem.MolFromSmiles(s)) for s in positive_samples]

    similarity_scores = []
    for fp in unl_fps:
        sims = [DataStructs.TanimotoSimilarity(fp, pfp) for pfp in pos_fps]
        similarity_scores.append(np.mean(sims))

    similarity_scores = np.array(similarity_scores)

    top_idx = np.argsort(similarity_scores)[-args.num_select:]
    selected_samples = [unlabeled_samples[i] for i in top_idx]

    morgan_loader = DataLoader(selected_samples, batch_size=args.pcl_batch_size, shuffle=False)

    return morgan_loader


if __name__ == '__main__':
    if not os.path.exists(f"./{args.save_path}"):
        os.makedirs(f"./{args.save_path}")

    ### load data
    print("Loading data...")
    positive_samples_126, unlabeled_samples, pos_train_samples, pos_test_samples, unl_train_samples, unl_test_samples = load_data(args)
    all_pos_samples = pos_train_samples + pos_test_samples
    pos_loader = DataLoader(pos_train_samples + pos_test_samples, batch_size=args.pcl_batch_size, shuffle=False)
    unl_loader = DataLoader(unl_train_samples + unl_test_samples, batch_size=args.pcl_batch_size, shuffle=False)
    print('Loaded data: Positive samples:', len(positive_samples_126), 'Unlabeled samples:', len(unlabeled_samples))



    ### recommendation method selection and model training
    if args.method == 'random':
        print('Random recommendation is selected. No model will be trained.')
        unl_loader = random_recommendation(unlabeled_samples, args.random_state)
        pos_loader = None

    elif args.method == 'morgan':
        print('Morgan fingerprint-based recommendation is selected. No model will be trained.')
        unl_loader = morgan_recommendation(positive_samples_126, unlabeled_samples)
        pos_loader = None

    elif args.method == 'full_model':
        print('Model-based recommendation is selected. Start training the models and doing recommendation.')
        # training the unsupervised learning model (USL)
        print("Getting the representation model...")
        usl = USL(args)
        usl_encoder = usl.get_representation_model(pos_train_samples, pos_test_samples)


        # training the prototype contrastive learning model (PCL)
        proto_train_samples = pos_train_samples + unl_train_samples
        proto_test_samples = pos_test_samples + unl_test_samples
        pcl = PCL(args, all_pos_samples)
        total_best_encoders, total_best_projections, total_best_embeddings, total_best_labels, total_best_proto_centroids = pcl.pcl_training(usl_encoder, proto_train_samples, proto_test_samples)

        torch.save(total_best_encoders.state_dict(), f'{args.save_path}/PCL_encoder_{args.method}_ema_{args.EMA}_decor_{args.use_decor_loss}_topk_{args.use_topk}.pth')
        torch.save(total_best_projections.state_dict(), f'{args.save_path}/PCL_projection_{args.method}_ema_{args.EMA}_decor_{args.use_decor_loss}_topk_{args.use_topk}.pth')
        torch.save(total_best_proto_centroids, f'{args.save_path}/proto_centroids.pth')


    else:
        raise Exception('Invalid recommendation method. Please choose either "random" or "full_model".')
    



    ### do the molecular recommendation
    print('\nStart the recommendation process...')
    recommender = Recommender(args, pos_loader, unl_loader)
    emb_pos, pos_labels, pos_ids, pos_sc_score, emb_unl, unl_labels, unl_ids, unl_sc_score, predict_labels_df = do_recommendation(recommender, pos_loader, unl_loader)




    ### do the post-screening
    print('\nStart the post-screening process...')
    postscreener = PostScreening(args, predict_labels_df)
    unl_ids = predict_labels_df['id'].tolist()
    samples_after_post_screening_df = postscreener.filter(emb_unl, unl_ids)

