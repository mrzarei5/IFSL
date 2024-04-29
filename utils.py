import argparse


def parse_args(network):
    parser = argparse.ArgumentParser(description='FSL script %s' %(network))
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers for datasets')
    parser.add_argument('--dataset_dir', default='', help='dataset location')
    parser.add_argument('--dataset', default='CUB', help='dataset name')
    
    
    if network == 'att_predictor':
        parser.add_argument('--n_iter', type = int, default=100, help='Number of training epochs')
        parser.add_argument('--lr_backbone_network', type= float, default=0.001, help='learning rate for backbone network')
        parser.add_argument('--save_dir_predictor', default='./checkpoints', help='Save directory root location for attribute predictor')
        
    elif network == 'att_selector':
        parser.add_argument('--n_episode_train', type = int, default=500, help='Number of episodes in each training iteration')
        parser.add_argument('--n_iter', type = int, default=400, help='Number of training iterations')
        parser.add_argument('--alpha', type=float, default=1, help='alpha')
        parser.add_argument('--gamma', type= float, default=0.005, help='gamma')
        parser.add_argument('--lr_att_selector', type= float, default=0.001, help='learning rate for attribute selector network')
        parser.add_argument('--save_dir_selector', default='./checkpoints', help='Save directory root location for attribute selector')
        parser.add_argument('--n_way', type= int, default=5, help='Number of classes in each episode')
        parser.add_argument('--n_query', type= int, default=16, help='Number of query samples for each class')
        parser.add_argument('--n_support', type= int, default=5, help='Number of support samples in each class')
        parser.add_argument('--tau', type= float, default=4, help='Temperature of Gumbel Softmax')
        parser.add_argument('--save_dir_predictor', default='./checkpoints', help='Save directory root location for attribute predictor')

    elif network == 'att_predictor_unk':
        parser.add_argument('--n_episode_train', type = int, default=500, help='Number of episodes in each training iteration')
        parser.add_argument('--n_iter', type = int, default=400, help='Number of training iterations')
        parser.add_argument('--lr_backbone_network', type= float, default=0.001, help='learning rate for backbone network')
        parser.add_argument('--save_dir_predictor_unknown', default='./checkpoints', help='Save directory root location for unknown attribute predictor')
        parser.add_argument('--n_way', type= int, default=5, help='Number of classes in each episode')
        parser.add_argument('--n_query', type= int, default=16, help='Number of query samples for each class')
        parser.add_argument('--n_support', type= int, default=1, help='Number of support samples in each class')
        parser.add_argument('--save_dir_predictor', default='./checkpoints', help='Save directory root location for attribute predictor')
        parser.add_argument('--lr_mi_helper', type = float, default=0.01, help='learning rate for mi helper')
        parser.add_argument('--n_mi_learner', type = int, default=10, help='Number of episodes for training mi learner in each iteration')
        parser.add_argument('--decoupling_weight', type = float, default=2.0, help='Decoupling weight')

    elif network == 'unk_att_participation':
        parser.add_argument('--n_episode_train', type = int, default=500, help='Number of episodes in each training iteration')
        parser.add_argument('--n_iter', type = int, default=400, help='Number of training iterations')
        parser.add_argument('--alpha', type=float, default=1, help='alpha')
        parser.add_argument('--beta', type= float, default=0.005, help='beta')
        parser.add_argument('--gamma', type= float, default=0, help='gamma')
        parser.add_argument('--lr_unk_part_detector', type= float, default=0.001, help='learning rate for unknown attribute participation network')
        parser.add_argument('--save_dir_unknown_participation', default='./checkpoints', help='Save directory root location for unknown attributes participation detector')
        parser.add_argument('--n_way', type= int, default=5, help='Number of classes in each episode')
        parser.add_argument('--n_query', type= int, default=16, help='Number of query samples for each class')
        parser.add_argument('--n_support', type= int, default=1, help='Number of support samples in each class')
        parser.add_argument('--tau', type= float, default=4, help='Temperature of Gumbel Softmax')

    elif network == 'dataset':
        parser.add_argument('--n_way', type= int, default=5, help='Number of classes in each episode')
        parser.add_argument('--n_query', type= int, default=16, help='Number of query samples for each class')
        parser.add_argument('--n_support', type= int, default=1, help='Number of support samples in each class')
        parser.add_argument('--save_dir', default='./checkpoints', help='Save directory root')

    elif network == 'evaluation':
        parser.add_argument('--n_way', type= int, default=5, help='Number of classes in each episode')
        parser.add_argument('--n_query', type= int, default=16, help='Number of query samples for each class')
        parser.add_argument('--n_support', type= int, default=1, help='Number of support samples in each class')
        parser.add_argument('--save_dir', default='./checkpoints', help='Save directory root location')
        parser.add_argument('--alpha', type=float, default=1, help='alpha')
        parser.add_argument('--beta', type= float, default=0.005, help='beta')
        parser.add_argument('--gamma', type= float, default=0, help='gamma')
        parser.add_argument('--tau', type= float, default=4, help='Temperature of Gumbel Softmax')

    elif network == 'intervention':
        parser.add_argument('--n_way', type= int, default=5, help='Number of classes in each episode')
        parser.add_argument('--n_query', type= int, default=16, help='Number of query samples for each class')
        parser.add_argument('--n_support', type= int, default=1, help='Number of support samples in each class')
        parser.add_argument('--save_dir', default='./checkpoints', help='Save directory root location')
        parser.add_argument('--alpha', type=float, default=1, help='alpha')
        parser.add_argument('--beta', type= float, default=0.005, help='beta')
        parser.add_argument('--gamma', type= float, default=0, help='gamma')
        parser.add_argument('--tau', type= float, default=0.5, help='Temperature of Gumbel Softmax')
    return parser.parse_args()