import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import AttSelector, AttPredictor, AttSelectorUnknown
from utils import parse_args, setup_logger
from datasets import FSLDataset
from config import dataset_config

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UnknownParticipationDetectorTrainer:
    """
    Trainer for detecting unknown attribute participation
    in few-shot episodes.
    """

    def __init__(self, args):
        self.args = args
        self.device = device

        # Hyperparameters
        self.lr = args.lr_unk_part_detector
        self.n_iter = args.n_iter
        self.num_workers = args.num_workers
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.tau = args.tau
        self.n_way = args.n_way
        self.n_support = args.n_support
        self.n_query = args.n_query
        self.n_episode_train = args.n_episode_train
        self.n_episode_test = 600  # fixed as in paper

        # Dataset configuration
        self.dataset = args.dataset
        self.dataset_dir = args.dataset_dir

        # Prepare paths and directories
        self._setup_paths()

        # Prepare data loaders
        self._setup_data_loaders()

        # Prepare networks
        self._setup_networks()


        # Set up separate loggers for epoch metrics and best-model events
        perf_log = os.path.join(self.save_dir, 'model_perf.txt')
        best_log = os.path.join(self.save_dir, 'best_perf.txt')
        self.logger_perf = setup_logger("PerfLogger", perf_log, console=True)
        self.logger_best = setup_logger("BestLogger", best_log)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = optim.Adam(self.att_selector_unknown.parameters(), lr=self.lr)

        # Freeze pretrained models
        for param in self.predictor_known.parameters():
            param.requires_grad = False
        for param in self.predictor_unknown.parameters():
            param.requires_grad = False
        for param in self.att_selector_known.parameters():
            param.requires_grad = False

        self.best_acc = 0.0


    def _setup_paths(self):
        base = self.args.save_dir_unknown_participation
        ds = self.dataset
        n_s = self.n_support
        a = self.alpha
        b = self.beta
        g = self.gamma

        # Pretrained models paths
        self.save_dir_att = os.path.join(base, ds)
        self.save_dir_att_fake = os.path.join(base, ds, f"{n_s}_shot_unknown", "n_mi_learner_10_decoupling_weight_2.0")
        self.save_dir_att_selector = os.path.join(base, ds, f"{self.n_way}_way_{n_s}_shot", f"l1_{a}_l2_{g}")

        # Directory for this trainer's outputs
        self.save_dir = os.path.join(base, ds, f"{self.n_way}_way_{n_s}_unknown_participation", f"l1_{a}_l2_{b}")
        os.makedirs(self.save_dir, exist_ok=True)

    def _setup_data_loaders(self):
        # Determine attribute sizes and network dimensions
        config = dataset_config[self.dataset]
        self.att_size = config['att_size']
        self.hid_dim_list = config['hid_dim_list']
        self.att_size_fake = config['att_size']
        self.hid_dim_list_fake = config['hid_dim_list']
        self.input_size = config['input_size']
        
        # Create few-shot dataset loaders
        train_ds = FSLDataset(
            self.dataset_dir, 'base', self.n_episode_train,
            self.n_way, self.n_support, self.n_query,
            aug=True, input_size=self.input_size
        )
        val_ds = FSLDataset(
            self.dataset_dir, 'val', self.n_episode_test,
            self.n_way, self.n_support, self.n_query,
            aug=False, input_size=self.input_size
        )
        test_ds = FSLDataset(
            self.dataset_dir, 'novel', self.n_episode_test,
            self.n_way, self.n_support, self.n_query,
            aug=False, input_size=self.input_size
        )

        self.train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=1, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )

    def _setup_networks(self):
        # Load pretrained attribute predictor (real)
        self.predictor_known = AttPredictor(
            hid_dim_list=self.hid_dim_list,
            att_size=self.att_size
        ).to(self.device)
        state_known = torch.load(os.path.join(self.save_dir_att, 'backboneNetwork-best.pth.tar'))
        self.predictor_known.load_state_dict(state_known['model_state_dict'])

        # Load pretrained attribute predictor (fake)
        self.predictor_unknown = AttPredictor(
            hid_dim_list=self.hid_dim_list_fake,
            att_size=self.att_size_fake
        ).to(self.device)
        state_unk = torch.load(os.path.join(self.save_dir_att_fake, 'backboneNetwork-best.pth.tar'))
        self.predictor_unknown.load_state_dict(state_unk['model_state_dict'])

        # Load real attribute selector
        self.att_selector_known = AttSelector(
            tau=self.tau,
            att_size=self.att_size
        ).to(self.device)
        sel_state = torch.load(os.path.join(self.save_dir_att_selector, 'attSelector-best.pth.tar'))
        self.att_selector_known.load_state_dict(sel_state['model_state_dict'])

        # Initialize unknown attribute selector
        self.att_selector_unknown = AttSelectorUnknown(
            tau=self.tau,
            att_size=self.att_size + self.att_size_fake,
            index_size=1
        ).to(self.device)



    @staticmethod
    def _compute_fsl_distances(prototypes, query_feats, prototypes_fake, query_feats_fake, sel_idx, sel_idx_real):
        """
        Compute combined squared distances for real and fake features.
        """
        # real embeddings
        qf = query_feats.contiguous().view(-1, query_feats.size(-1)).unsqueeze(1)
        pf = prototypes.unsqueeze(0)
        qf = qf.expand(-1, pf.size(1), -1)
        pf = pf.expand(qf.size(0), -1, -1)

        # fake embeddings
        qf_fake = query_feats_fake.contiguous().view(-1, query_feats_fake.size(-1)).unsqueeze(1)
        pf_fake = prototypes_fake.unsqueeze(0)
        qf_fake = qf_fake.expand(-1, pf_fake.size(1), -1)
        pf_fake = pf_fake.expand(qf_fake.size(0), -1, -1)

        dist_real = ((qf - pf) ** 2 * sel_idx_real).sum(2)
        dist_fake = ((qf_fake - pf_fake) ** 2 * sel_idx[0]).sum(2)
        return dist_real + dist_fake

    @staticmethod
    def _calculate_acc(prototypes, query_feats, prototypes_fake, query_feats_fake, sel_idx, sel_idx_real, n_query):
        """
        Compute classification accuracy for one episode.
        """
        dists = UnknownParticipationDetectorTrainer._compute_fsl_distances(
            prototypes, query_feats.unsqueeze(1),
            prototypes_fake, query_feats_fake.unsqueeze(1),
            sel_idx, sel_idx_real
        )
        preds = torch.argmin(dists, dim=1)
        labels = torch.arange(prototypes.size(0), device=preds.device)
        labels = labels.unsqueeze(1).repeat(1, n_query).view(-1)
        correct = (preds == labels).sum().item()
        return correct / labels.numel() * 100

    def evaluate(self, loader):
        """
        Evaluate model on given loader.
        Returns: (mean_acc, ci95, mean_sel, std_sel)
        """
        self.predictor_known.eval()
        self.predictor_unknown.eval()
        self.att_selector_unknown.eval()
        self.att_selector_known.eval()

        acc_list = []
        sel_list = []

        with torch.no_grad():
            for data in loader:
                imgs, _atts = data
                images = imgs[0].to(self.device)
                # reshape
                n_cls, n_ex, c, h, w = images.size()
                images_reshaped = images.contiguous().view(n_cls * n_ex, c, h, w)

                # predict attributes
                pred_real = self.predictor_known(images_reshaped)
                pred_fake = self.predictor_unknown(images_reshaped)
                pred_real = pred_real.contiguous().view(n_cls, self.n_support + self.n_query, -1)
                pred_fake = pred_fake.contiguous().view(n_cls, self.n_support + self.n_query, -1)

                # compute prototypes
                proto_real = pred_real[:, :self.n_support, :].mean(1).view(n_cls, -1)
                proto_fake = pred_fake[:, :self.n_support, :].mean(1).view(n_cls, -1)
                mixed_proto = torch.cat((proto_real, proto_fake), dim=1)

                # selection indices
                sel_idx, _ = self.att_selector_unknown(mixed_proto)
                sel_idx = sel_idx.view(-1)
                sel_idx_real, _ = self.att_selector_known(proto_real)

                sel_list.append(sel_idx.detach())

                # query features
                q_real = pred_real[:, self.n_support:, :].contiguous()
                q_fake = pred_fake[:, self.n_support:, :].contiguous()

                # compute accuracy
                acc = self._calculate_acc(
                    proto_real, q_real, proto_fake, q_fake,
                    sel_idx, sel_idx_real, self.n_query
                )
                acc_list.append(acc)

        sel_sums = [s.sum().item() for s in sel_list]
        acc_arr = np.array(acc_list)
        mean_acc = acc_arr.mean()
        ci95 = 1.96 * acc_arr.std() / np.sqrt(len(acc_arr))
        return mean_acc, ci95, np.mean(sel_sums), np.std(sel_sums)

    def train(self):
        """
        Main training loop.
        """
        for epoch in tqdm(range(self.n_iter), desc="Epoch"): 
            epoch_loss = 0.0
            loss_class = 0.0
            loss_sel = 0.0
            sel_list = []

            self.att_selector_unknown.train()
            for data in self.train_loader:
                imgs, _atts = data
                images = imgs[0].to(self.device)
                n_cls, n_ex, c, h, w = images.size()
                images_reshaped = images.contiguous().view(n_cls * n_ex, c, h, w)

                # attribute predictions
                pred_real = self.predictor_known(images_reshaped)
                pred_fake = self.predictor_unknown(images_reshaped)
                pred_real = pred_real.view(n_cls, self.n_support + self.n_query, -1)
                pred_fake = pred_fake.view(n_cls, self.n_support + self.n_query, -1)

                # prototypes
                proto_real = pred_real[:, :self.n_support, :].mean(1).view(n_cls, -1)
                proto_fake = pred_fake[:, :self.n_support, :].mean(1).view(n_cls, -1)
                mixed_proto = torch.cat((proto_real, proto_fake), dim=1)

                # selection
                sel_idx, _ = self.att_selector_unknown(mixed_proto)
                sel_idx = sel_idx.view(-1)
                sel_idx_real, _ = self.att_selector_known(proto_real)
                sel_list.append(sel_idx.detach())

                # compute distances and losses
                dists = self._compute_fsl_distances(
                    proto_real, pred_real[:, self.n_support:, :],
                    proto_fake, pred_fake[:, self.n_support:, :],
                    sel_idx, sel_idx_real
                )
                labels = torch.arange(n_cls, device=self.device)
                labels = labels.unsqueeze(1).repeat(1, self.n_query).view(-1)
                loss1 = self.criterion(-dists, labels)
                loss2 = sel_idx.sum()

                loss = self.alpha * loss1 + self.beta * loss2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                loss_class += (self.alpha * loss1).item()
                loss_sel += (self.beta * loss2).item()

            # Compute metrics after epoch
            train_sel_sums = [s.sum().item() for s in sel_list]
            val_acc, val_ci, val_mean, val_std = self.evaluate(self.val_loader)

            # Log performance

            self.logger_perf.info(
                f"Epoch {epoch} | "
                f"Loss {epoch_loss/len(self.train_loader):.4f} | "
                f"Class Loss {loss_class/len(self.train_loader):.4f} | "
                f"Select Loss {loss_sel/len(self.train_loader):.4f} | "
                f"Val Acc {val_acc:.2f} | CI95 {val_ci:.2f} | "
                f"Temp {self.att_selector_unknown.tau:.4f}"
            )
            self.logger_perf.info(
                f"Epoch {epoch} | "
                f"Train Sel Mean {np.mean(train_sel_sums):.2f} | "
                f"Train Sel Std {np.std(train_sel_sums):.2f}"
            )
            self.logger_perf.info(
                f"Epoch {epoch} | "
                f"Val Sel Mean {val_mean:.2f} | "
                f"Val Sel Std {val_std:.2f}\n"
            )

            # Save best model if criteria met
            if val_acc > self.best_acc and 0.1 <= val_mean <= 0.9:
                test_acc, test_ci, test_mean, test_std = self.evaluate(self.test_loader)
                self.logger_best.info(
                    f"Epoch {epoch} | "
                    f"Loss {epoch_loss/len(self.train_loader):.4f} | "
                    f"Class Loss {loss_class/len(self.train_loader):.4f} | "
                    f"Select Loss {loss_sel/len(self.train_loader):.4f} | "
                    f"Val Acc {val_acc:.2f} | CI95 {val_ci:.2f} | "
                    f"Test Acc {test_acc:.2f} | CI95 {test_ci:.2f} | "
                    f"Temp {self.att_selector_unknown.tau:.4f}"
                )
                self.logger_best.info(
                    f"Epoch {epoch} | "
                    f"Train Sel Mean {np.mean(train_sel_sums):.2f} | "
                    f"Train Sel Std {np.std(train_sel_sums):.2f}"
                )
                self.logger_best.info(
                    f"Epoch {epoch} | "
                    f"Val Sel Mean {val_mean:.2f} | "
                    f"Val Sel Std {val_std:.2f} | "
                    f"Test Sel Mean {test_mean:.2f} | Test Sel Std {test_std:.2f}\n"
                )
                # Save selector state
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.att_selector_unknown.state_dict(),
                    'val_acc': val_acc,
                    'train_loss': epoch_loss,
                    'best_acc': self.best_acc
                }, os.path.join(self.save_dir, 'attSelector-best.pth.tar'))
                self.best_acc = val_acc

            # Decay temperature
            if epoch > 0 and epoch % 25 == 0 and self.att_selector_unknown.tau > 0.5:
                self.att_selector_unknown.tau /= 2


def main():
    args = parse_args('unk_att_participation')
    trainer = UnknownParticipationDetectorTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
