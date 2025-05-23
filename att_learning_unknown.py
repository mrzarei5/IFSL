import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm

from utils import parse_args, setup_logger
from datasets import FSLDataset
from models import AttPredictor, MutualHelper
from config import dataset_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def estimate_mutual_information(mi_model, known_feats, unknown_feats, unknown_marginal_feats, eval_mode = 0):
    """
    Estimate mutual information lower bound between known and unknown features.

    Args:
        mi_model: the mutual information network.
        known_feats: features from the known attribute predictor.
        unknown_feats: features from the unknown attribute backbone.
        unknown_marginal_feats: additional unknown features for marginal estimation.
        eval_mode: flag to set the model to evaluation mode.

    Returns:
        The estimated mutual information lower bound.
    """
    if eval_mode == 1:
        mi_model.eval()

    joint = torch.cat((known_feats, unknown_feats), dim=1)
    marginal = torch.cat((known_feats, unknown_marginal_feats), dim=1)

    term1 = mi_model(joint).mean()
    term2 = torch.log(torch.exp(mi_model(marginal)).mean())
    return term1 - term2

def compute_fsl_distances(prototypes_features,query_features): 
    """
    Compute distance between query features and prototypes

    Args:
        prototypes: class prototype vectors.
        query_features: query feature vectors.

    Returns:
        Pairwise distance matrix between prototypes and queries.
    """
    query_features_ = torch.unsqueeze(query_features.view(query_features.size(0)*query_features.size(1),query_features.size(2)),1)
    prototypes_features_ = torch.unsqueeze(prototypes_features,0)

    query_features_ = query_features_.expand(-1,  prototypes_features_.size(1), -1)   
    
    prototypes_features_ = prototypes_features_.expand(query_features_.size(0), -1, -1)

    diff = (query_features_ - prototypes_features_)**2

    return diff.sum(2)


def compute_accuracy(prototypes,query_features, n_query): 
    """
    Computes accuracy for a given few-shot task.

    Args:
        prototypes: Prototype vectors for classes
        query_features: Query feature vectors
        selected_attribute_mask: Mask for selected attributes
        n_query: Number of query samples per class

    Returns:
        Accuracy as a percentage
    """

    query_features_ = torch.unsqueeze(query_features,1)
    prototypes_ = torch.unsqueeze(prototypes,0)

    query_features_ = query_features_.expand(-1,prototypes_.size(1),-1)   
    prototypes_ = prototypes_.expand(query_features_.size(0),-1,-1)

    dist  = (((query_features_- prototypes_)**2)).sum(2)

    min_args = torch.argmin(dist,1)

    query_labels = torch.tensor([[i] for i in range(prototypes.size(0))]).to(device)
    query_labels = query_labels.repeat(1,n_query).flatten()    
    correct = (min_args == query_labels).sum().item()
    return correct / len(query_labels) * 100


def prepare_data(args):
    """
    Instantiate train, validation, test, and MI DataLoaders.

    Args:
        args: runtime arguments containing dataset and loader settings.

    Returns:
        Four data loaders for training, validation, testing, and MI training.
    """
    ds_dir = args.dataset_dir
    way, support, query = args.n_way, args.n_support, args.n_query
    workers = args.num_workers
    size = args.input_size


    def get_loader(split, n_episode, aug=False):
        dataset = FSLDataset(ds_dir, split, n_episode, way, support, query, aug, size)
        return DataLoader(dataset, batch_size=1, shuffle=aug, num_workers= workers, pin_memory=True)

    return (
        get_loader('base', args.n_episode_train, True),
        get_loader('val', args.n_episode_test, False),
        get_loader('novel', args.n_episode_test, False),
        get_loader('base', args.n_mi_learner, True)
    )

def setup_models(args, device):
    """
    Load pretrained predictor (known), instantiate unknown predictor and MI networks, and their optimizers.

    Args:
        args: runtime arguments containing model and optimizer settings.
        device: computation device.

    Returns:
        Known predictor, unknown predictor, MI network, and their optimizers.
    """
    att_size = args.att_size
    hid_dim_list = args.hid_dim_list

    # Known attribute predictor (frozen)
    known_predictor = AttPredictor(
        hid_dim_list=hid_dim_list,
        att_size=att_size
    )
    ckpt = torch.load(os.path.join(args.save_dir_predictor, args.dataset, 'backboneNetwork-best.pth.tar'))
    known_predictor.load_state_dict(ckpt['model_state_dict'])
    known_predictor.to(device).eval()
    for param in known_predictor.parameters():
        param.requires_grad = False
    
    # Unknown attribute predictor to train
    unknown_predictor = AttPredictor(
        hid_dim_list=hid_dim_list,
        att_size=att_size
    ).to(device)
    optimizer_unknown = optim.Adam(unknown_predictor.parameters(), lr=args.lr_backbone_network)

    # Mutual information network
    mi_network = MutualHelper(att_size_overall= (att_size + att_size)).to(device)
    optimizer_mi = optim.Adam(mi_network.parameters(), lr=args.lr_mi_helper)

    return known_predictor, unknown_predictor, mi_network, optimizer_unknown, optimizer_mi



class FewShotTrainer:
    def __init__(self, args):
        self.args = args
        self.device = device
        self.train_loader, \
        self.val_loader,   \
        self.test_loader,  \
        self.mi_loader      = prepare_data(args)

        (self.known_predictor,
         self.unknown_predictor,
         self.mi_network,
         self.optimizer_unknown,
         self.optimizer_mi) = setup_models(args, device)

        self.criterion = nn.CrossEntropyLoss()
        self.best_acc  = 0.0

        self.save_dir = os.path.join(args.save_dir_predictor_unknown, 
                                     args.dataset,
                                     f"{args.n_support}_shot_unknown",
                                     f"n_mi_learner_{args.n_mi_learner}_decoupling_weight_{args.decoupling_weight}")
        os.makedirs(self.save_dir, exist_ok=True)


        # Initialize loggers
        perf_path = os.path.join(self.save_dir, 'model_perf.txt')
        best_path = os.path.join(self.save_dir, 'best_perf.txt')
        self.logger_perf = setup_logger("PerfLogger", perf_path, console=True)
        self.logger_best = setup_logger("BestLogger", best_path)

    def train_mutual_network(self, images, real_feats):
        """
        One pass of MI-network training over the MI loader.
        """
        for p in self.unknown_predictor.parameters():
            p.requires_grad = False
        for p in self.mi_network.parameters():
            p.requires_grad = True

        fake_feats1 = self.unknown_predictor(images)

        self.mi_network.train()
        for data in self.mi_loader:
            images2, _ = data
            images2     = images2[0].to(self.device)

            images_size = images2.size()

            images2 = images2.contiguous().view(images_size[0]*images_size[1],images_size[2],images_size[3],images_size[4])

      
            fake_feats2 = self.unknown_predictor(images2)

            loss_mi = -estimate_mutual_information(
                self.mi_network,
                real_feats,
                fake_feats1,
                fake_feats2
            )

            self.mi_network.zero_grad()
            loss_mi.backward(retain_graph=True)
            self.optimizer_mi.step()

    def train_epoch(self, epoch):
        """
        Runs one full epoch of backbone + MI-regularized training.
        """
        self.unknown_predictor.train()
        total_loss = total_class = total_mi = total_mival = 0.0

        for data in self.train_loader:
            images, _ = data
            images     = images[0].to(self.device)
            images_size = images.size()

            images = images.view(images_size[0]*images_size[1],images_size[2],images_size[3],images_size[4])

            real_feats = self.known_predictor(images)
            self.train_mutual_network(images, real_feats)

            for p in self.mi_network.parameters():
                p.requires_grad = False
            for p in self.unknown_predictor.parameters():
                p.requires_grad = True

            fake_feats = self.unknown_predictor(images)
            
            episodes   = fake_feats.contiguous().view(self.args.n_way,self.args.n_support + self.args.n_query,-1)
            
            support_feats    = episodes[:, :self.args.n_support, :].mean(1).view(self.args.n_way,self.args.att_size_fake)
            query_feats = episodes[:, self.args.n_support:, :].contiguous().view(self.args.n_way,self.args.n_query, self.args.att_size_fake)

            dists   = compute_fsl_distances(support_feats, query_feats)
            labels  = torch.arange(self.args.n_way, device=self.device)\
                            .unsqueeze(1).repeat(1, self.args.n_query).view(-1)
            loss_ce = self.criterion(-dists, labels)

            # one MI validation sample
            data2, _ = next(iter(self.mi_loader))
            images2    = data2[0].to(self.device)
            n2, m2, c2, h2, w2 = images2.size()
            images2   = images2.view(n2*m2, c2, h2, w2)
            fake_feats2    = self.unknown_predictor(images2)

            mi_val   = estimate_mutual_information(
                self.mi_network,
                real_feats,
                fake_feats,
                fake_feats2,
                eval_mode=1
            )
            loss_mi  = self.args.decoupling_weight * mi_val

            loss = loss_ce + loss_mi
            self.unknown_predictor.zero_grad()
            loss.backward()
            self.optimizer_unknown.step()

            total_loss  += loss.item()
            total_class += loss_ce.item()
            total_mi    += loss_mi.item()
            total_mival += mi_val.item()

        return total_loss, total_class, total_mi, total_mival

    def evaluate(self, split='val'):
        """
        Evaluate few-shot accuracy over val/test loader.
        """
        loader = self.val_loader if split == 'val' else self.test_loader
        self.unknown_predictor.eval()
        accs = []

        with torch.no_grad():
            for data in loader:
                images, _ = data
                images     = images[0].to(self.device)
                n, m, c, h, w = images.size()
                flat     = images.contiguous().contiguous().view(n*m, c, h, w)

                feats    = self.unknown_predictor(flat)
                episodes = feats.view(n, m, -1)
                
                prototypes  = episodes[:, :self.args.n_support, :].mean(1).view(-1,self.args.att_size_fake)
                query_feats = episodes[:, self.args.n_support:, :].contiguous().view(n*self.args.n_query,self.args.att_size_fake)

                accs.append(compute_accuracy(prototypes, query_feats, self.args.n_query))

        arr  = np.array(accs)
        mean = arr.mean()
        ci95 = 1.96 * arr.std() / np.sqrt(len(arr))
        return mean, ci95

    def save_checkpoint(self, epoch, val_acc, val_std):
        """
        Save best-so-far model weights and log performance.
        """
        test_acc, test_std = self.evaluate('test')

        self.logger_best.info(
            f"Iter {epoch} | Val Acc {val_acc:.4f}±{val_std:.4f} | "
            f"Test Acc {test_acc:.4f}±{test_std:.4f}"
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.unknown_predictor.state_dict(),
            'val_acc': val_acc
        }, os.path.join(self.save_dir, 'backboneNetwork-best.pth.tar'))

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.mi_network.state_dict(),
            'val_acc': val_acc
        }, os.path.join(self.save_dir, 'mutualNetwork-best.pth.tar'))

    def run(self):
        """
        Main training loop.
        """
        perf_path = os.path.join(self.save_dir, 'model_perf.txt')
        for epoch in tqdm(range(self.args.n_iter)):
            tl, cl, ml, mv = self.train_epoch(epoch)
            val_acc, val_std = self.evaluate('val')

            self.logger_perf.info(
                f"Iter {epoch} | Loss {tl/len(self.train_loader):.4f} | "
                f"Class {cl/len(self.train_loader):.4f} | MI {ml/len(self.train_loader):.4f} | "
                f"MI Val {mv/len(self.train_loader):.4f} | "
                f"Val Acc {val_acc:.4f}±{val_std:.4f}"
            )


            with open(perf_path, 'a') as f:
                f.write(
                    f"Iter {epoch} | Loss {tl/len(self.train_loader):.4f} "
                    f"| Class {cl/len(self.train_loader):.4f} "
                    f"| MI {ml/len(self.train_loader):.4f} "
                    f"| MI Val {mv/len(self.train_loader):.4f} "
                    f"| Val Acc {val_acc:.4f}±{val_std:.4f}\n"
                )

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(epoch, val_acc, val_std)

        self.logger_best.info(f"Training complete. Best Val Acc: {self.best_acc:.2f}%")




def main():
    random.seed(10)
    args    = parse_args('att_predictor_unk')

    cfg = dataset_config[args.dataset]
    args.att_size     = cfg['att_size']
    args.hid_dim_list = cfg['hid_dim_list']
    args.input_size   = cfg['input_size']
    args.att_size_fake = args.att_size
    args.n_episode_test = 600

    trainer = FewShotTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
