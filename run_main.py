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
from utils.tools import mol_to_fp, extract_embeddings, select_by_cluster_centers
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from model import ProjectionHead

warnings.filterwarnings('ignore')


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description="Train the model")
# basic configs
parser.add_argument('--method', type=str, default='full_model', help='recommendation method')
parser.add_argument('--data_path', type=str, default='./data/all_data_year.pkl', help='Path to the preprocessed data')
parser.add_argument('--save_path', type=str, default='checkpoints_origin_backup', help='')
parser.add_argument('--device', type=str, default='cuda:7' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
parser.add_argument('--searching_space_path', type=str, default='./data/searching_space_data_V2.csv', help='Path to the CSV file containing the searching space data')
parser.add_argument('--additive_json_path', type=str, default='./data/additives_year_sorted.json', help='Path to the JSON file containing additive data')
parser.add_argument('--year_mapping_path', type=str, default='./data/year_split_mapping.json', help='Path to the JSON file containing year mapping data')
parser.add_argument('--split_year', type=str, default='all', help='Year to split the data')


# baseline configs
parser.add_argument('--num_select', type=int, default=30800 , help='Number of molecules to select')
parser.add_argument('--encoder_similarity', type=str, default='max', choices=['max', 'mean'], help='Similarity aggregation for encoder-only baselines: max or mean over positive samples.')
parser.add_argument('--cluster_num', type=int, default=7, help='Number of clusters for unsupervised_clustering baseline.')


# unsupervised learning configs
parser.add_argument('--analysis', type=str2bool, default=False, help='Wether to print the summary of the dataset')
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
parser.add_argument('--retrain_usl', type=str2bool, default=False, help='retrain the usl models')
parser.add_argument('--usl_trials', type=int, default=10, help='Number of trials for unsupervised learning')
parser.add_argument('--usl_backbone', type=str, default='GINE', help='Backbone GNN model for USL. Options: GINE, GAT')


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
parser.add_argument('--save_proto_drift', type=str2bool, default=False, help='Whether to save prototype drift information')
parser.add_argument('--EMA', type=str2bool, default=True, help='Whether to activate Exponential Moving Average (EMA) for prototype updating')
parser.add_argument('--use_decor_loss', type=str2bool, default=True, help='Whether to activate decorrelation loss for prototype learning')
parser.add_argument('--use_topk', type=str2bool, default=True, help='Whether to activate top-k selection for prototype learning')


# post-screening configs
parser.add_argument('--save_molecules', type=str2bool, default=True, help='Whether to save the recommended molecules after post-screening')
# parser.add_argument('--best_prototype_path', type=str, default='./result_files/proto_table_trial_7.csv', help='Path to the CSV file containing the best prototypes')
parser.add_argument('--post_screening_output_path', type=str, default='./outputs/', help='Path to the post-screening output file')
parser.add_argument('--recommend_model', type=str, default='full_model', help='')


args = parser.parse_args()

# random recommendation baseline
def random_recommendation(unlabeled_samples, random_state):
    # random recommend samples
    random.seed(random_state)
    recommended_samples = random.sample(unlabeled_samples, args.num_select)

    random_loader = DataLoader(recommended_samples, batch_size=args.pcl_batch_size, shuffle=False)

    return random_loader

# morgan fingerprint recomendation baseline
def morgan_recommendation(positive_samples, unlabeled_samples, unl_samples_graph):
    unl_fps = [mol_to_fp(Chem.MolFromSmiles(s['smiles'])) for s in tqdm(unlabeled_samples) if Chem.MolFromSmiles(s['smiles']) is not None]
    pos_fps = [mol_to_fp(Chem.MolFromSmiles(s)) for s in tqdm(positive_samples) if Chem.MolFromSmiles(s) is not None]

    similarity_scores = []
    for fp in tqdm(unl_fps, desc='Calculating the tanimoto similarity...'):
        sims = [DataStructs.TanimotoSimilarity(fp, pfp) for pfp in pos_fps]
        similarity_scores.append(np.mean(sims))

    similarity_scores = np.array(similarity_scores)

    top_idx = np.argsort(similarity_scores)[-args.num_select:]
    selected_samples = [unlabeled_samples[i] for i in top_idx]
    selected_ids = {s['id'] for s in selected_samples}
    recommend_unl_samples = [sample for sample in tqdm(unl_samples_graph) if sample.id in selected_ids]

    morgan_loader = DataLoader(recommend_unl_samples, batch_size=args.pcl_batch_size, shuffle=False)
    
    return morgan_loader

# encoder-only recommendation baseline
def encoder_only_positive_similarity_recommendation(args, pos_samples, unl_samples, encoder_type='pcl'):
    """
    Encoder-only positive similarity baseline.

    encoder_type='usl':
        Use USL encoder only, no PCL and no prototypes.

    encoder_type='pcl':
        Use pretrained PCL encoder + projection, but recommendation does NOT use prototypes.
        It only ranks candidates by similarity to positive-sample embeddings.

    Recommendation score:
        max_i cosine(z_unl, z_pos_i)  if args.encoder_similarity == 'max'
        mean_i cosine(z_unl, z_pos_i) if args.encoder_similarity == 'mean'
    """
    pos_loader = DataLoader(pos_samples, batch_size=args.pcl_batch_size, shuffle=False)
    unl_loader = DataLoader(unl_samples, batch_size=args.pcl_batch_size, shuffle=False)

    if encoder_type == 'usl':
        usl = USL(args)
        usl_encoder = usl.get_representation_model(pos_samples, [])
        encoder = usl_encoder.to(args.device)
        projection = ProjectionHead(in_dim=args.usl_hidden_channels).to(args.device)
        print('Using USL encoder-only positive similarity baseline.')

    elif encoder_type == 'pcl':
        pcl = PCL(args, pos_samples)

        if pcl.is_trained is False:
            raise Exception('PCL model not trained yet. Please train the PCL model before running the encoder-only baseline with PCL encoder.')
        else:
            encoder, projection = pcl.load_pcl_encoder_and_projection(pos_samples[0])

        print('Using PCL encoder-only positive similarity baseline.')

    else:
        raise ValueError(f'Unknown encoder_type: {encoder_type}')

    emb_pos, pos_ids = extract_embeddings(
        encoder=encoder,
        dataloader=pos_loader,
        device=args.device,
        projection=projection,
        normalize_embedding=True
    )

    emb_unl_all, unl_ids_all = extract_embeddings(
        encoder=encoder,
        dataloader=unl_loader,
        device=args.device,
        projection=projection,
        normalize_embedding=True
    )

    # cosine similarity because embeddings are normalized
    sim_matrix = emb_unl_all @ emb_pos.T

    if args.encoder_similarity == 'max':
        scores = sim_matrix.max(dim=1).values
    elif args.encoder_similarity == 'mean':
        scores = sim_matrix.mean(dim=1)
    else:
        raise ValueError(f'Unknown encoder_similarity: {args.encoder_similarity}')

    num_select = min(args.num_select, scores.numel())
    top_idx = torch.topk(scores, k=num_select).indices

    selected_emb_unl = emb_unl_all[top_idx]
    selected_unl_ids = unl_ids_all[top_idx]
    selected_scores = scores[top_idx]

    # label = -1 means no prototype label is assigned.
    selected_labels = (-1) * torch.ones(selected_unl_ids.size(0), dtype=torch.long)

    predict_labels_df = pd.DataFrame({
        'id': selected_unl_ids.cpu().numpy(),
        'label': selected_labels.cpu().numpy(),
        'encoder_similarity_score': selected_scores.cpu().numpy(),
    })

    print(f'{encoder_type.upper()} encoder-only selected molecules:', len(predict_labels_df))
    print('Selected similarity score range:',
          float(selected_scores.min()), 'to', float(selected_scores.max()))

    return selected_emb_unl, selected_labels, selected_unl_ids, None, predict_labels_df

# unsupervised clustering recommendation baseline
def unsupervised_clustering_recommendation(args, pos_samples, unl_samples, encoder_type='pcl'):
    """
    Unsupervised clustering baseline.

    It does not use positive-derived prototypes in the recommendation step.
    It clusters unlabeled embeddings and selects representative molecules near cluster centers.

    args.cluster_encoder='usl':
        Use USL encoder.

    args.cluster_encoder='pcl':
        Use PCL encoder + projection, but no positive-derived prototype selection.
    """

    pos_loader = DataLoader(pos_samples, batch_size=args.pcl_batch_size, shuffle=False)
    unl_loader = DataLoader(unl_samples, batch_size=args.pcl_batch_size, shuffle=False)

    if encoder_type == 'usl':
        usl = USL(args)
        usl_encoder = usl.get_representation_model(pos_samples, [])
        encoder = usl_encoder.to(args.device)
        projection = ProjectionHead(in_dim=args.usl_hidden_channels).to(args.device)
        print('Using USL encoder for unsupervised clustering baseline.')

    elif encoder_type == 'pcl':
        pcl = PCL(args, pos_samples)

        if pcl.is_trained is False:
            raise Exception('PCL model not trained yet. Please train the PCL model before running the encoder-only baseline with PCL encoder.')
        else:
            encoder, projection = pcl.load_pcl_encoder_and_projection(pos_samples[0])

        print('Using PCL encoder for unsupervised clustering baseline.')

    else:
        raise ValueError(f'Unknown cluster_encoder: {encoder_type}')

    emb_unl_all, unl_ids_all = extract_embeddings(
        encoder=encoder,
        dataloader=unl_loader,
        device=args.device,
        projection=projection,
        normalize_embedding=True
    )

    X = emb_unl_all.numpy()
    cluster_num = min(args.cluster_num, len(X))

    print(f'Running MiniBatchKMeans with n_clusters={cluster_num}...')
    kmeans = MiniBatchKMeans(
        n_clusters=cluster_num,
        random_state=args.random_state,
        batch_size=max(1024, args.pcl_batch_size),
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(X)
    selected_idx = select_by_cluster_centers(
        embeddings=emb_unl_all,
        ids=unl_ids_all,
        cluster_labels=cluster_labels,
        cluster_centers=kmeans.cluster_centers_,
        num_select=min(args.num_select, len(X))
    )

    selected_emb_unl = emb_unl_all[selected_idx]
    selected_unl_ids = unl_ids_all[selected_idx]

    selected_labels = torch.tensor(
        cluster_labels[selected_idx.numpy()],
        dtype=torch.long
    ) + 1

    predict_labels_df = pd.DataFrame({
        'id': selected_unl_ids.cpu().numpy(),
        'label': selected_labels.cpu().numpy(),
        'cluster_label': selected_labels.cpu().numpy(),
    })

    print('Unsupervised clustering selected molecules:', len(predict_labels_df))
    print('Cluster distribution among selected molecules:')
    print(predict_labels_df['cluster_label'].value_counts().sort_index())

    return selected_emb_unl, selected_labels, selected_unl_ids, None, predict_labels_df



if __name__ == '__main__':
    if not os.path.exists(f"./{args.save_path}"):
        os.makedirs(f"./{args.save_path}")


    ## recommendation method selection and model training
    if args.method == 'random':
        print("Loading data...")
        _, unlabeled_samples, _, _, _, _ = load_data(args)

        print('Random recommendation is selected. No model will be trained.')
        unl_loader = random_recommendation(unlabeled_samples, args.random_state)
        pos_loader = None

    elif args.method == 'morgan':
        print("Loading data...")
        positive_samples, unlabeled_samples, unl_samples_graph = load_data(args)

        print('Morgan fingerprint-based recommendation is selected. No model will be trained.')
        unl_loader = morgan_recommendation(positive_samples, unlabeled_samples, unl_samples_graph)
        pos_loader = None

    elif args.method == 'full_model':
        ### load data
        print("Loading data...")
        positive_samples_126, unlabeled_samples, pos_train_samples, pos_test_samples, unl_train_samples, unl_test_samples = load_data(args)

        print('Model-based recommendation is selected. Start training the models and doing recommendation.')
        print("Getting the representation model...")

        all_pos_samples = pos_train_samples + pos_test_samples
        pos_loader = DataLoader(pos_train_samples + pos_test_samples, batch_size=args.pcl_batch_size, shuffle=False)
        unl_loader = DataLoader(unl_train_samples + unl_test_samples, batch_size=args.pcl_batch_size, shuffle=False)
        print('Loaded data: Positive samples:', len(positive_samples_126), 'Unlabeled samples:', len(unlabeled_samples))

        # training the unsupervised learning model (USL)
        usl = USL(args)
        usl_encoder = usl.get_representation_model(pos_train_samples, pos_test_samples)


        # training the prototype contrastive learning model (PCL)
        proto_train_samples = pos_train_samples + unl_train_samples
        proto_test_samples = pos_test_samples + unl_test_samples
        pcl = PCL(args, all_pos_samples)

        if pcl.is_trained is False:
            ### train the PCL model
            total_best_encoders, total_best_projections, total_best_embeddings, total_best_labels, total_best_proto_centroids = pcl.pcl_training(usl_encoder, proto_train_samples, proto_test_samples)
            # save the best PCL model
            torch.save(total_best_encoders.state_dict(), f'{args.save_path}/PCL_encoder_{args.method}_ema_{args.EMA}_decor_{args.use_decor_loss}_topk_{args.use_topk}_year_{args.split_year}.pth')
            torch.save(total_best_projections.state_dict(), f'{args.save_path}/PCL_projection_{args.method}_ema_{args.EMA}_decor_{args.use_decor_loss}_topk_{args.use_topk}_year_{args.split_year}.pth')
            torch.save(total_best_proto_centroids, f'{args.save_path}/proto_centroids_{args.method}_ema_{args.EMA}_decor_{args.use_decor_loss}_topk_{args.use_topk}_year_{args.split_year}.pth')
        else:
            print('Pretrained PCL model detected. Loading the model and doing recommendation...')
    
    elif args.method == 'usl_encoder_only':
        print("Loading data...")
        positive_samples_126, unlabeled_samples, pos_train_samples, pos_test_samples, unl_train_samples, unl_test_samples = load_data(args)

        all_pos_samples = pos_train_samples + pos_test_samples
        all_unl_samples = unl_train_samples + unl_test_samples

        emb_unl, unl_labels, unl_ids, unl_sc_score, predict_labels_df = encoder_only_positive_similarity_recommendation(
            args=args,
            pos_samples=all_pos_samples,
            unl_samples=all_unl_samples,
            encoder_type='usl'
        )

    elif args.method == 'pcl_encoder_only':
        print("Loading data...")
        positive_samples_126, unlabeled_samples, pos_train_samples, pos_test_samples, unl_train_samples, unl_test_samples = load_data(args)

        all_pos_samples = pos_train_samples + pos_test_samples
        all_unl_samples = unl_train_samples + unl_test_samples

        emb_unl, unl_labels, unl_ids, unl_sc_score, predict_labels_df = encoder_only_positive_similarity_recommendation(
            args=args,
            pos_samples=all_pos_samples,
            unl_samples=all_unl_samples,
            encoder_type='pcl'
        )

    elif args.method == 'usl_encoder_clustering':
        print("Loading data...")
        positive_samples_126, unlabeled_samples, pos_train_samples, pos_test_samples, unl_train_samples, unl_test_samples = load_data(args)

        all_pos_samples = pos_train_samples + pos_test_samples
        all_unl_samples = unl_train_samples + unl_test_samples

        emb_unl, unl_labels, unl_ids, unl_sc_score, predict_labels_df = unsupervised_clustering_recommendation(
            args=args,
            pos_samples=all_pos_samples,
            unl_samples=all_unl_samples,
            encoder_type='usl'
        )

    elif args.method == 'pcl_encoder_clustering':
        print("Loading data...")
        positive_samples_126, unlabeled_samples, pos_train_samples, pos_test_samples, unl_train_samples, unl_test_samples = load_data(args)

        all_pos_samples = pos_train_samples + pos_test_samples
        all_unl_samples = unl_train_samples + unl_test_samples

        emb_unl, unl_labels, unl_ids, unl_sc_score, predict_labels_df = unsupervised_clustering_recommendation(
            args=args,
            pos_samples=all_pos_samples,
            unl_samples=all_unl_samples,
            encoder_type='pcl'
        )

    else:
        raise Exception('Invalid recommendation method.')





    ### do the molecular recommendation
    if args.method in ['usl_encoder_only', 'pcl_encoder_only', 'usl_encoder_clustering', 'pcl_encoder_clustering']:
        print('Encoder-only positive similarity recommendation is done. No need to do the rest of the recommendation process.')
    else:
        print('\nStart the recommendation process...')
        recommender = Recommender(args, pos_loader, unl_loader)
        emb_pos, pos_labels, pos_ids, pos_sc_score, emb_unl, unl_labels, unl_ids, unl_sc_score, predict_labels_df = do_recommendation(recommender, pos_loader, unl_loader)




    ### do the post-screening
    print('\nStart the post-screening process...')
    postscreener = PostScreening(args, predict_labels_df)
    unl_ids = predict_labels_df['id'].tolist()
    samples_after_post_screening_df = postscreener.filter(emb_unl, unl_ids)

