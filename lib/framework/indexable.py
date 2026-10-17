R"""
"""
#
import abc
import numpy as onp
import torch
import time
import os
from typing import List, Tuple, Optional, Generic, TypeVar
from ..meta.meta import Meta
from .transfer import transfer
from .types import TIMECOST
from ..task.task import Task
from ..utils.info import info5
from .gradclip import GRADCLIPS
from tqdm import tqdm
import wandb
from ..task.classification import METRICS_REP
#
META = TypeVar("META", bound=Meta)
from ..exec.consts import DIR, LR_DECAY, LR_THRES, IMP_ABS, IMP_REL, CLF_TASK


class FrameworkIndexable(abc.ABC, Generic[META]):
    R"""
    Framework with indexable (by finite integer) meta samples.
    """
    #
    BATCH_PAD: bool

    def __init__(
        self,
        identifier: str, metaset: META, neuralnet: Task,
        /,
        *,
        lr: float, weight_decay: float, seed: int, device: str,
        metaspindle: str, gradclip: str,
    ) -> None:
        R"""
        Initalize the class.
        """
        #
        self.identifier = identifier
        self.metaset = metaset
        self.neuralnet = neuralnet
        self.seed = seed
        self.device = device

        #
        self.metaspindle = metaspindle
        self.lr = lr
        self.weight_decay = weight_decay
        #
        if neuralnet.num_resetted_params > 0:
            #
            self.gradclip = GRADCLIPS[gradclip]
            self.optim = (
                torch.optim.Adam(
                    self.neuralnet.parameters(),
                    lr=lr, weight_decay=weight_decay,
                )
            )
            # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            #     self.optim,
            #     T_0=50,
            #     T_mult=1,
            #     eta_min=LR_THRES,
            # )

        # Move model to device after creating the optimizer.
        self.neuralnet = neuralnet.to(device)

        # Ensure the existence of log directory.
        if not os.path.isdir(DIR):
            #
            os.makedirs(DIR, exist_ok=True)
        self.ptnnp = os.path.join(DIR, "{:s}.ptnnp".format(self.identifier))
        self.ptres = os.path.join(DIR, "{:s}.ptres".format(self.identifier))
        self.ptlog = os.path.join(DIR, "{:s}.ptlog".format(self.identifier))

    @abc.abstractmethod
    def train(
        self,
        meta_indices: List[int], meta_index_pad: Optional[int],
        meta_batch_size: int, pinned: List[torch.Tensor],
        /,
    ) -> TIMECOST:
        R"""
        Train.
        Mostly used for neural network parameter tuning.
        """
        #
        ...

    @abc.abstractmethod
    def evaluate(
        self,
        meta_indices: List[int], meta_index_pad: Optional[int],
        meta_batch_size: int, pinned: List[torch.Tensor],
        /,
    ) -> Tuple[List[float], TIMECOST]:
        R"""
        Evaluate.
        Mostly used for neural network parameter evaluation.
        """
        #
        ...

    def preprocess(
        self,
        proportion: Tuple[int, int, int], priority: Tuple[int, int, int],
        train_prop: Tuple[int, int, bool],
        /,
    ) -> None:
        R"""
        Preprocess metaset before fitting.
        """
        # Split and normalize.
        print("=" * 10 + " " + "(Prep)rocessing" + " " + "=" * 10)
        (
            self.meta_indices_train, self.meta_indices_valid,
            self.meta_indices_test,
        ) = (
            self.metaset.fitsplit(proportion, priority, self.metaspindle)
            if train_prop[1] == 0 else
            self.metaset.reducesplit(
                proportion, priority, self.metaspindle, *train_prop,
            )
        )
        meta_size_train = len(self.meta_indices_train)
        meta_size_valid = len(self.meta_indices_valid)
        meta_size_test = len(self.meta_indices_test)
        meta_size = meta_size_train + meta_size_valid + meta_size_test

        self.factors = (
            self.metaset.normalizeby(self.meta_indices_train, self.metaspindle)
        )
        #
        print(
            info5(
                {
                    "Split": {
                        "Train": (
                            "{:d}/{:d}".format(meta_size_train, meta_size)
                        ),
                        "Validate": (
                            "{:d}/{:d}".format(meta_size_valid, meta_size)
                        ),
                        "Test": (
                            "{:d}/{:d}".format(meta_size_test, meta_size)
                        ),
                    },
                },
            )
        )
        print(self.metaset.distrep(n=3))

    def fit(
        self,
        proportion: Tuple[int, int, int], priority: Tuple[int, int, int],
        train_prop: Tuple[int, int, bool],
        /,
        *,
        batch_size: int, max_epochs: int, validon: int, validrep: str,
        patience: int, min_epochs: int = 100,
    ) -> None:
        R"""
        Fit neural network of the framework based on initialization status.
        """
        USE_LR_SCHEDULER = False
        #
        self.preprocess(proportion, priority, train_prop)
        
        split_sizes = {
            'train': len(self.meta_indices_train),
            'valid': len(self.meta_indices_valid),
            'test': len(self.meta_indices_test),
        }
        wandb.config.update({'split_sizes': split_sizes})
        totalsize = sum(split_sizes.values())
        split_sizes_perc = {k: round(v / totalsize * 100) for k, v in split_sizes.items()}
        wandb.config.update({'split_sizes_perc': split_sizes_perc})

        def print_label_counts(indices: onp.ndarray, axis: int, split_name: str) -> None:
            unique_labels = onp.unique(self.metaset.node_labels)
            lb_cnts = onp.zeros(len(unique_labels), dtype=int)
            split_labels = onp.take(self.metaset.node_labels, indices, axis=axis)
            for i, ul in enumerate(unique_labels):
                lb_cnts[i] = onp.sum(split_labels == ul)

            lb_cnts = onp.array(lb_cnts)
            print(f'label counts {split_name}: {lb_cnts} ({[f"{v:.2f}%" for v in lb_cnts / lb_cnts.sum()]})')
        
        if self.metaspindle == 'node' and self.metaset.task_type == CLF_TASK:
            dyn_label = self.metaset.dyn_node_labels
            print_label_counts(self.meta_indices_train, 0, 'trainset')
            print_label_counts(self.meta_indices_valid, 0, 'validset')
            print_label_counts(self.meta_indices_test, 0, 'testset')
        elif self.metaspindle == 'time' and self.metaset.task_type == CLF_TASK:
            dyn_label = self.metaset.dyn_node_labels
            print_label_counts(self.meta_indices_train, 2, 'trainset')
            print_label_counts(self.meta_indices_valid, 2, 'validset')
            print_label_counts(self.meta_indices_test, 2, 'testset')
        #
        # Getting LR scheduler
        if not USE_LR_SCHEDULER:
            self.lr_scheduler = None
            print("No LR scheduler is used.")
            wandb.config.update({
                'lr_scheduler': {
                    'type': 'None',
                }
            })
        else:
            #
            print(
                f"Using ReduceLROnPlateau LR scheduler with factor 1/{LR_DECAY} "
                f"and threshold {LR_THRES:.1e}.",
            )
            patience = max_epochs * 0.1 if patience < 0 else patience
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optim,
                mode="min", # metrics are negated if higher is better
                factor=1 / LR_DECAY,
                patience=patience,
                threshold=1e-6,
                verbose=True,
            )
            num_warmup_epochs = 0
            wandb.config.update({  
                'lr_scheduler': {
                    'type': 'ReduceLROnPlateau',
                    'factor': 1 / LR_DECAY,
                    'patience': patience,
                    'stop_threshold': LR_THRES
                }
            })
        #
        timecosts: TIMECOST

        # Zero-out number of tuning epochs for non-parametric cases.
        if self.neuralnet.num_resetted_params == 0:
            #
            max_epochs = 0
        
        #
        rng = onp.random.RandomState(self.seed)
        timecosts = {}
        meta_index_pad = (
            onp.min(self.meta_indices_train).item() if self.BATCH_PAD else None
        )
        epochlen = len(str(max_epochs))
        logs = []
        metric_best = 0.0
        best_epoch = 0
        num_not_improving = 0
        noimplen = len(str(patience))

        # Pin shared memory.
        elapsed = time.time()
        split_sizes 
        pinned_numpy = self.metaset.pin(batch_size)
        timecosts["pin.generate"] = time.time() - elapsed
        elapsed = time.time()
        pinned_ondev = transfer(pinned_numpy, self.device)
        timecosts["pin.transfer"] = time.time() - elapsed

        # Validate once before training.
        print("=" * 10 + " " + "Train & Validate" + " " + "=" * 10)
        with torch.no_grad():
            #
            (metrics, timeparts) = (
                self.evaluate(
                    self.meta_indices_valid.tolist(), meta_index_pad,
                    batch_size, pinned_ondev
                )
            )
        timecosts["valid.generate"] = timeparts["generate"]
        timecosts["valid.transfer"] = timeparts["transfer"]
        timecosts["valid.forward"] = timeparts["forward"]
        logs.append(metrics)

        # Initialize performance status.
        torch.save(self.neuralnet.state_dict(), self.ptnnp)
        nn_classname = self.neuralnet.tgnn.__class__.__name__
        
        metric_best = metrics[validon]
        num_not_improving = 0
        print(
            "[{:>{:d}d}/{:d}] {:s}: {:>8s} ({:>8s}) {:s}{:<{:d}s}".format(
                0, epochlen, max_epochs, f"({nn_classname}) {validrep}",
                "{:.6f}".format(metrics[validon])[:8],
                "{:.6f}".format(metric_best)[:8],
                "\x1b[92m↑\x1b[0m", "", noimplen,
            ),
        )

        #
        timecosts["train.generate"] = []
        timecosts["train.transfer"] = []
        timecosts["train.forward"] = []
        timecosts["train.backward"] = []
        for epoch in tqdm(range(1, 1 + max_epochs)):
            #
            shuffling = rng.permutation(len(self.meta_indices_train))
            meta_indices_train = self.meta_indices_train[shuffling]

            # Train.
            timeparts, epoch_train_loss = (
                self.train(
                    meta_indices_train.tolist(), meta_index_pad, batch_size,
                    pinned_ondev,
                )
            )
            timecosts["train.generate"] = timeparts["generate"]
            timecosts["train.transfer"] = timeparts["transfer"]
            timecosts["train.forward"] = timeparts["forward"]
            timecosts["train.backward"] = timeparts["backward"]

            # Validate.
            with torch.no_grad():
                #
                (metrics, timeparts) = (
                    self.evaluate(
                        self.meta_indices_valid.tolist(), meta_index_pad,
                        batch_size, pinned_ondev,
                    )
                )
            timecosts["valid.generate"] = timeparts["generate"]
            timecosts["valid.transfer"] = timeparts["transfer"]
            timecosts["valid.forward"] = timeparts["forward"]
            logs.append(metrics)
            # Apply changes to the model states after evaluat
            # Update performance status
            if (
                metrics[validon] < metric_best - IMP_ABS
                or metrics[validon] < metric_best * (1 - IMP_REL)
            ):
                #
                torch.save(self.neuralnet.state_dict(), self.ptnnp)
                metric_best = metrics[validon]
                best_epoch = epoch
                improving = True
                num_not_improving = 0
            else:
                #
                improving = False
                num_not_improving = num_not_improving + 1
            print(
                "[{:>{:d}d}/{:d}] {:s}: {:>8s} ({:>8s}) {:s}{:<{:d}s}{:s}"
                .format(
                    epoch, epochlen, max_epochs, f"({nn_classname}) {validrep}",
                    "{:.6f}".format(metrics[validon])[:8],
                    "{:.6f}".format(metric_best)[:8],
                    "\x1b[92m↑\x1b[0m" if improving else "\x1b[91m↓\x1b[0m",
                    "" if improving else str(num_not_improving),
                    noimplen, " --" if num_not_improving == patience else "",
                ),
            )
            
            # Log to wandb.
            log_obj = {
                f"fit_{validrep}": metrics[validon],
                f"fit_best_val_{validrep}": metric_best,
                "improving": improving,
                "train_loss": epoch_train_loss,
                "train_generate_time": sum(timecosts["train.generate"]),
                "train_forward_time": sum(timecosts["train.forward"]),
                "train_transfer_time": sum(timecosts["train.transfer"]),
                "train_backward_time": sum(timecosts["train.backward"]),
                "valid_generate_time": timecosts["valid.generate"],
                "valid_forward_time": timecosts["valid.forward"],
                "valid_transfer_time": timecosts["valid.transfer"],
                "num_not_improving": num_not_improving,
                "learning_rate": self.optim.param_groups[0]["lr"],
            }
            wandb.log(log_obj, step=epoch, commit=True)

            # LR scheduler step and Early stopping check.
            if self.lr_scheduler is None:
                pass
            elif isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(metrics[validon])
            else:
                self.lr_scheduler.step()
            
            if USE_LR_SCHEDULER:
                # Reset num_not_improving if still in warmup phase.
                if epoch < num_warmup_epochs:
                    print("  (In warmup phase, reset num_not_improving to 0)")
                    num_not_improving = 0
            # Early stopping.
            if patience >= 0 and num_not_improving > patience:
                #
                if epoch < min_epochs:
                    #
                    print(
                        "Continue training until epoch {:d} due to min_epochs={:d}"
                        .format(min_epochs, min_epochs),
                    )
                    num_not_improving = patience
                else:
                    #
                    print(
                        "Early stopping at epoch {:d} with {:s}: {:.6f}"
                        .format(epoch, validrep, metric_best),
                    )
                    break
            #
            if not USE_LR_SCHEDULER:
                if patience >= 0 :
                    lr_reach_epsilon = False
                    if self.neuralnet.num_resetted_params > 0:
                        for group in self.optim.param_groups:
                            if group["lr"] < LR_THRES:
                                lr_reach_epsilon = True
                                break
                    else:
                        lr_reach_epsilon = True
                    if lr_reach_epsilon:
                        # Early stop if learning rate is too small.
                        break
                    
        if hasattr(self.neuralnet, "tgnn"):
            #
            torch.save(
                (
                    torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
                    getattr(getattr(self.neuralnet, "tgnn"), "COSTS"),
                ),
                self.ptres,
            )

        # Final test after training.
        print("=" * 10 + " " + "Test" + " " + "=" * 10)
        with torch.no_grad():
            #
            (metrics_valid, timeparts) = (
                self.evaluate(
                    self.meta_indices_valid.tolist(), meta_index_pad,
                    batch_size, pinned_ondev,
                )
            )
            print(
                "Valid\x1b[94m:\x1b[0m \x1b[3m{:s}\x1b[0m: {:s}"
                .format(validrep, "{:.6f}".format(metrics_valid[validon])[:8]),
            )
            # Test only if there are test samples.
            metrics_test = {}
            if len(self.meta_indices_test) > 0:
                (metrics_test, timeparts) = (
                    self.evaluate(
                        self.meta_indices_test.tolist(), meta_index_pad,
                        batch_size, pinned_ondev,
                    )
                )
                timecosts["test.generate"] = timeparts["generate"]
                timecosts["test.transfer"] = timeparts["transfer"]
                timecosts["test.forward"] = timeparts["forward"]
                
                print(
                    " Test\x1b[94m:\x1b[0m \x1b[3m{:s}\x1b[0m: {:s}"
                    .format(validrep, "{:.6f}".format(metrics_test[validon])[:8]),
                )
                
        # Log to wandb.
        # Log final test metrics to wandb summary for easy access.
        wandb.run.summary[f"fittest_val_{validrep}"] = metrics_valid[validon]
        # log other metrics
        for i, metric in enumerate(metrics_valid):
            wandb.run.summary[f"fittest_val_{METRICS_REP[i]}"] = metric
        
        if len(self.meta_indices_test) > 0:
            wandb.run.summary["fittest_test"] = metrics_test[validon]
            for i, metric in enumerate(metrics_test):
                wandb.run.summary[f"fittest_test_{METRICS_REP[i]}"] = metric

        wandb.run.summary["bestval_epoch"] = best_epoch
        #
        print("=" * 10 + " " + "(Res)ource (Stat)istics" + " " + "=" * 10)
        gpu_mem_peak = int(onp.ceil(torch.cuda.max_memory_allocated() / 1024))
        print("Max GPU Memory: {:d} KB".format(gpu_mem_peak))
        wandb.run.summary["gpu_mem_peak_kb"] = gpu_mem_peak
        torch.save(
            (
                self.factors, logs, metrics_valid[validon], metrics_test,
                gpu_mem_peak, timecosts
            ),
            self.ptlog,
        )

    def besteval(
        self,
        proportion: Tuple[int, int, int], priority: Tuple[int, int, int],
        train_prop: Tuple[int, int, bool],
        /,
        batch_size: int, validon: int, validrep: str, resume: str,
    ) -> None:
        R"""
        Best test after training.
        """
        #
        self.ptnnp = os.path.join(resume, "{:s}.ptnnp".format(self.identifier))
        self.ptlog = os.path.join(resume, "{:s}.ptlog".format(self.identifier))

        # Ensure the existence of log directory.
        if not os.path.isdir(DIR):
            #
            os.makedirs(DIR, exist_ok=True)
        self.ptbev = os.path.join(DIR, "{:s}.ptbev".format(self.identifier))

        #
        self.preprocess(proportion, priority, train_prop)

        #
        meta_index_pad = (
            onp.min(self.meta_indices_train).item() if self.BATCH_PAD else None
        )

        # Pin shared memory.
        pinned_numpy = self.metaset.pin(batch_size)
        pinned_ondev = transfer(pinned_numpy, self.device)

        # Best test after training.
        print("=" * 10 + " " + "Test (best)" + " " + "=" * 10)
        self.neuralnet.load_state_dict(torch.load(self.ptnnp, weights_only=False), strict=False)
        
        with torch.no_grad():
            #
            (metrics_valid, _) = (
                self.evaluate(
                    self.meta_indices_valid.tolist(), meta_index_pad,
                    batch_size, pinned_ondev,
                )
            )
            (metrics_test, _) = (
                self.evaluate(
                    self.meta_indices_test.tolist(), meta_index_pad,
                    batch_size, pinned_ondev,
                )
            )
        print(
            "Valid\x1b[94m:\x1b[0m \x1b[3m{:s}\x1b[0m: {:s}"
            .format(validrep, "{:.6f}".format(metrics_valid[validon])[:8]),
        )
        print(
            " Test\x1b[94m:\x1b[0m \x1b[3m{:s}\x1b[0m: {:s}"
            .format(validrep, "{:.6f}".format(metrics_test[validon])[:8]),
        )

        #
        print("=" * 10 + " " + "Relog" + " " + "=" * 10)
        (factors, logs, _, _, gpu_mem_peak, timecosts) = torch.load(self.ptlog, weights_only=False)
        torch.save(
            (
                factors, logs, metrics_valid[validon], metrics_test,
                gpu_mem_peak, timecosts,
            ),
            self.ptbev,
        )
        
        # Log to wandb.
        wandb.run.summary[f"besteval_val_{validrep}"] = metrics_valid[validon]
        wandb.run.summary["besteval_test"] = metrics_test[validon]
        # log other metrics
        for i, metric in enumerate(metrics_valid):
            wandb.run.summary[f"besteval_val_{METRICS_REP[i]}"] = metric
        for i, metric in enumerate(metrics_test):
            wandb.run.summary[f"besteval_test_{METRICS_REP[i]}"] = metric
