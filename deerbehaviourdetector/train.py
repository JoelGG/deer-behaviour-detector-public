from data.split_dataset import SplitDataset
from data.wr_sampler import balanced_wrsampler
from models.single_stream_models import DeerSingleStreamResnet, DeerSingleStreamMViT
from models.two_stream_models import (
    TwoStreamModelAdaptiveFuse,
    TwoStreamModelAvgFuse,
    TwoStreamModelStackFuse,
)
from evaluation.logger import LogWriter
from torchvision.models.video import R3D_18_Weights, MViT_V2_S_Weights

import time
import gc
from typing import Union
from sklearn.metrics import confusion_matrix
import argparse
from pathlib import Path
from multiprocessing import cpu_count
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


parser = argparse.ArgumentParser(
    description="Train model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--video-root",
    default=Path("/user/work/ki19061/dataset"),
    type=Path,
    help="Path to a directory containing all data-set input information",
)
parser.add_argument(
    "--annotation-root",
    default=Path(
        "/user/work/ki19061/deer-behaviour-detector/behaviour_annotations_2.csv"
    ),
    type=Path,
    help="Path to a csv file containing annotations for the video data-set",
)
parser.add_argument("-k", "--kinetic", action="store_true")
parser.add_argument("-b", "--balanced-sampling", action="store_true")
parser.add_argument(
    "--log-dir",
    default=Path("tensorboard_logs"),
    type=Path,
    help="Directory to save tensorboard logs",
)
parser.add_argument(
    "--fig-dir",
    default=Path("figs"),
    type=Path,
    help="Directory to save figures during training, such as confusion matrices",
)
parser.add_argument(
    "--learning-rate", default=0.00005, type=float, help="Learning rate"
)
parser.add_argument("--l2-alpha", default=1e-5, type=float, help="L2 alpha")


parser.add_argument(
    "--batch-size",
    default=128,
    type=int,
    help="Number of videos within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=200,
    type=int,
    help="Number of epochs to train for",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="Frequency at which to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="Frequency at which to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="Frequency at which to print progress to the command line in number of steps",
)

parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)


def cuda_metrics():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved
    print(f"Total: {t}, Reserved: {r}, Allocated: {a}, Free: {f}")


def report_gpu():
    print(torch.cuda.list_gpu_processes())
    cuda_metrics()
    gc.collect()
    torch.cuda.empty_cache()
    cuda_metrics()


torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.version.cuda)
print(DEVICE)
if DEVICE == torch.device("cuda"):
    report_gpu()


def main(args):
    frame_transform = default_transform(R3D_18_Weights.DEFAULT.transforms())
    optical_flow_transform = flow_transform(size=(112, 112))

    train_dataset = SplitDataset(
        args.video_root,
        "/user/work/ki19061/deer-behaviour-detector/deerbehaviourdetector/annotations/annotations_train.csv",
        frame_transform,
        optical_flow_transform,
        cache_name="resnet_train",
    )
    test_dataset = SplitDataset(
        args.video_root,
        "/user/work/ki19061/deer-behaviour-detector/deerbehaviourdetector/annotations/annotations_test.csv",
        frame_transform,
        optical_flow_transform,
        cache_name="resnet_test",
    )

    if args.balanced_sampling:
        sampler = balanced_wrsampler(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=args.worker_count,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=args.worker_count,
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    num_classes = 2 if args.kinetic else 8

    model = DeerSingleStreamResnet(
        num_classes=num_classes, feature_extract=False, pretrained=False
    )
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.l2_alpha,
    )

    log_dir = get_summary_writer_log_dir(args)
    fig_dir = get_summary_writer_fig_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(str(log_dir), flush_secs=5)
    log_writer = LogWriter(
        itercount=(args.epochs * len(train_dataset)), logdir="logfiles/run1.csv"
    )

    start_epoch = 0

    trainer = Trainer(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        summary_writer,
        DEVICE,
        fig_dir,
        args.kinetic,
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        log_writer,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
        start_epoch=start_epoch,
        l2_alpha=args.l2_alpha,
    )

    summary_writer.close()


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
        fig_dir: Path,
        kinetic: bool,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.fig_dir = fig_dir
        self.step = 0
        self.kinetic = kinetic

    def train(
        self,
        epochs: int,
        val_frequency: int,
        log_writer,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0,
        l2_alpha=0.0001,
    ):
        early_stopping_tracker = {
            "best_val_loss": np.inf,
            "epochs_since_improvement": 0,
            "patience": 10,
        }
        accumulation_steps = 4

        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for i, (batch) in enumerate(self.train_loader):
                video = batch["video"]
                labels = batch["target"]
                kinetic_labels = batch["kinetic_target"]
                if self.kinetic:
                    labels = kinetic_labels
                batch_rgb_cropped = video["rgb_cropped"].to(self.device)
                batch_flow_cropped = video["flow_cropped"].to(self.device)

                labels = labels.to(self.device)
                data_load_end_time = time.time()

                logits = self.model(batch_rgb_cropped, batch_flow_cropped)
                loss = self.criterion(logits, labels)
                loss = loss / accumulation_steps
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    with torch.no_grad():
                        preds = logits.argmax(-1)
                        accuracy = compute_accuracy(labels, preds)

                    data_load_time = data_load_end_time - data_load_start_time
                    step_time = time.time() - data_load_end_time
                    if ((self.step + 1) % log_frequency) == 0:
                        self.log_metrics(
                            epoch, accuracy, loss, data_load_time, step_time
                        )
                        self.save_metrics(
                            log_writer,
                            self.step % len(self.train_loader),
                            accuracy,
                            loss,
                            "train",
                        )
                    if ((self.step + 1) % print_frequency) == 0:
                        self.print_metrics(
                            epoch, accuracy, loss, data_load_time, step_time
                        )

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)

            if ((epoch + 1) % val_frequency) == 0:
                save_confusion = ((epoch + 1) % 10) == 0
                self.validate(save_confusion=save_confusion, log_writer=log_writer)
                log_writer.to_csv()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
            f"epoch: [{epoch}], "
            f"step: [{epoch_step}/{len(self.train_loader)}], "
            f"batch loss: {loss:.5f}, "
            f"batch accuracy: {accuracy * 100:2.2f}, "
            f"data load time: "
            f"{data_load_time:.5f}, "
            f"step time: {step_time:.5f}"
        )

    def save_metrics(self, log_writer, epoch, accuracy, loss, t="train"):
        log_writer.add_scalars(f"loss {t}", float(loss.item()), self.step)
        log_writer.add_scalars(f"top1 {t}", f"{accuracy * 100:2.2f}", self.step)

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars("accuracy", {"train": accuracy}, self.step)
        self.summary_writer.add_scalars(
            "loss", {"train": float(loss.item())}, self.step
        )
        self.summary_writer.add_scalar("time/data", data_load_time, self.step)
        self.summary_writer.add_scalar("time/data", step_time, self.step)

    def validate(
        self, save_confusion=False, early_stopping_tracker=None, log_writer=None
    ):
        results = {
            "preds": [],
            "logits": [],
            "kinetic_labels": [],
            "labels": [],
            "video_ids": [],
            "starts": [],
            "ends": [],
        }
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch in self.val_loader:
                video = batch["video"]
                labels = batch["target"]
                kinetic_labels = batch["kinetic_target"]
                video_id = batch["video_id"]
                start = batch["start"]
                end = batch["end"]
                if self.kinetic:
                    labels = kinetic_labels

                batch_rgb_cropped = video["rgb_cropped"].to(self.device)
                batch_flow_cropped = video["flow_cropped"].to(self.device)

                labels = labels.to(self.device)
                logits = self.model(batch_rgb_cropped, batch_flow_cropped)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()

                results["preds"].extend(list(preds))
                results["logits"].extend(logits.cpu().numpy().tolist())
                results["labels"].extend(list(labels.cpu().numpy()))
                results["video_ids"].extend(list(video_id))
                results["starts"].extend(list(start))
                results["ends"].extend(list(end))
        classes = [
            "walk_left",
            "walk_right",
            "walk_towards",
            "walk_away",
            "standing",
            "grooming",
            "browsing",
            "ear_scratching",
        ]
        classes.sort()
        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        all_class_accuracies, average_class_accuray = compute_all_class_accuracy(
            np.array(results["labels"]),
            np.array(results["preds"]),
            range(0, len(classes)),
        )
        top_k_accuracies = top_k_accuracy(
            torch.Tensor(results["logits"]), torch.Tensor(results["labels"]), topk=(3,)
        )
        avg_t_k = torch.stack(top_k_accuracies).mean()
        print(f"Top3: {avg_t_k}")
        print(all_class_accuracies)
        print(average_class_accuray)
        # raw_accuracy = evalutation.evaluate(preds, "data/val.pkl", self.device)
        average_loss = total_loss / len(self.val_loader)

        if save_confusion:
            c_matrix = confusion_matrix(
                results["labels"], results["preds"], labels=range(0, len(classes))
            )
            print(c_matrix)
            df_c_matrix = pd.DataFrame(
                c_matrix / np.sum(c_matrix, axis=1)[:, None],
                index=[i for i in range(0, len(classes))],
            )
            print(df_c_matrix)
            plt.figure(figsize=(12, 7))
            sn.heatmap(
                df_c_matrix.iloc[:, 0:2] if self.kinetic else df_c_matrix,
                annot=True,
                xticklabels=["Static", "Kinetic"] if self.kinetic else classes,
                yticklabels=classes,
                vmin=0,
                vmax=1,
                square=True,
            )
            plt.savefig(f"{self.fig_dir}_{self.step}.png")

        self.summary_writer.add_scalars("accuracy", {"test": accuracy}, self.step)
        # self.summary_writer.add_scalars("raw_accuracy", {"test": raw_accuracy}, self.step)
        self.summary_writer.add_scalars("loss", {"test": average_loss}, self.step)
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")
        if early_stopping_tracker:
            if average_loss < early_stopping_tracker.best_loss:
                early_stopping_tracker.best_loss = average_loss
                early_stopping_tracker.epochs_since_improvement = 0
            else:
                early_stopping_tracker.epochs_since_improvement += 1
                if early_stopping_tracker.epochs_since_improvement > 10:
                    print("Early stopping")
                    return True


def extract_matches(labels, preds, video_ids, starts, ends):
    labels = np.array(labels)
    preds = np.array(preds)
    output = np.array(list(zip(video_ids, starts, ends)))
    hits = output[labels == preds]
    misses = output[labels != preds]
    return hits, misses


def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


# INPUTS: output have shape of [batch_size, category_count]
#    and target in the shape of [batch_size] * there is only one true class for each sample
# topk is tuple of classes to be included in the precision
# topk have to a tuple so if you are giving one number, do not forget the comma
def top_k_accuracy(output, target, topk=(3,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # we do not need gradient calculation for those
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()
        target_reshaped = target.view(1, -1).expand_as(y_pred)
        correct = y_pred == target_reshaped

        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = (
                ind_which_topk_matched_truth.reshape(-1).float()
            )  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(
                dim=0, keepdim=True
            )  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]


def compute_class_accuracy(
    labels: Union[torch.Tensor, np.ndarray],
    preds: Union[torch.Tensor, np.ndarray],
    class_index: int,
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
        class_index ``class_index`` index of class for accuracy in range 0-9
    """
    assert len(labels) == len(preds)
    idx = labels == class_index
    return float((labels[idx] == preds[idx]).sum() / len(labels[idx]))


def compute_all_class_accuracy(
    labels: Union[torch.Tensor, np.ndarray],
    preds: Union[torch.Tensor, np.ndarray],
    class_list,
) -> np.ndarray:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
        class_count: ``class_count`` total number of classes
    """
    assert len(labels) == len(preds)
    accuracies = np.zeros((len(class_list)))
    for c in range(0, len(class_list)):
        accuracies[c] = compute_class_accuracy(labels, preds, c)
    return accuracies, accuracies.mean()


def mixup_data(x, y, device, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def dir_prefix(args):
    return (
        f"CNN_bn_"
        f"bs={args.batch_size}_"
        f"lr={args.learning_rate}_"
        f"l2_alpha={args.l2_alpha}_"
        f"run_"
    )


def get_checkpoint_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = dir_prefix(args)
    i = 0
    while i < 1000:
        tb_log_dir = args.checkpoint_path / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = dir_prefix(args)
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


def get_summary_writer_fig_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been figged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of fig_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_fig_dir_prefix = dir_prefix(args)
    i = 0
    while i < 1000:
        tb_fig_dir = args.fig_dir / (tb_fig_dir_prefix + str(i))
        if not tb_fig_dir.exists():
            return str(tb_fig_dir)
        i += 1
    return str(tb_fig_dir)


class CheckpointWriter:
    def __init__(self, path):
        self.path = Path(path)

    def write(self, epoch, model_state, optim_state, loss):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
            },
            self.path,
        )


def default_transform(preprocess):
    def transform_t(t: torch.Tensor) -> torch.Tensor:
        t = transforms.ToTensor()(t)
        t = preprocess(t.unsqueeze(0))
        t = t.squeeze()
        return t

    return transform_t


def flow_transform(size):
    def transform_t(t: torch.Tensor) -> torch.Tensor:
        t = transforms.ToTensor()(t)
        t = transforms.Resize(size, antialias=None)(t)
        t = t.squeeze()
        return t

    return transform_t


if __name__ == "__main__":
    main(parser.parse_args())
