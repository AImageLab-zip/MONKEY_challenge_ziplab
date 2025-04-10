import argparse
import inspect
import io
import os
import pdb
import subprocess

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from Classifiers import PatchClassifier
from instanseg.utils.augmentations import Augmentations
from instanseg.utils.utils import show_images
from kornia.augmentation import (
    ColorJiggle,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)
from pytorch_lightning.callbacks import DeviceStatsMonitor, TQDMProgressBar
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms import ToTensor


def describe_batch(x):
    print(x.shape, x.dtype)
    print(x[:, :3].min(), x[:, :3].max())
    print(x[:, :3].min(), x[:, :3].max())
    print(x[:, :3].mean(), x[:, :3].std())


def my_collate_fn(batch):
    dataset_id = torch.tensor([item[-1] for item in batch])

    images = torch.stack([i[0] for i in batch])
    labels = [i[1] for i in batch]

    return images, labels, dataset_id


class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, split="train", apply_color_jitter: bool = True) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter

        if split == "train":
            self.transforms = nn.Sequential(
                # Resize(224),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                # RandomThinPlateSpline(p=0.5),
                # kornia.augmentation.RandomRotation(degrees = 90.0,p=0.5),
            )

            self.jitter = ColorJiggle(0.2, 0.2, 0.1, 0.1)

        if split == "val":
            self.transforms = nn.Sequential(
                # Resize(224),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
            )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x):
        x_out = self.transforms(x)  # BxCxHxW
        if self._apply_color_jitter:
            if x_out.shape[1] == 4:
                x_out[:, :3] = self.jitter(x_out[:, :3])

        return x_out


def get_gpu_utilization():
    try:
        # Run `nvidia-smi` and parse output
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        # Parse the utilization values (handles multiple GPUs)
        gpu_util = [int(x) for x in result.strip().split("\n")]
        return gpu_util
    except Exception as e:
        print(f"Error fetching GPU utilization: {e}")
        return []


class PatchClassifier_pl(pl.LightningModule):
    def __init__(
        self,
        num_classes=16,
        embedding_dim=64,
        pretrained=True,
        l_fns="cross_entropy",
        encoder="mobilenet_v3",
        from_file=None,
        class_names=None,
        dim_in=4,
        batch_size=128,
        num_epochs=256,
        len_epoch=100000,
        weight_decay=1e-4,
        dropprob=0.1,
        accumulate_grad_batches=1,
        jitter=True,
        data_name=["none"],
        data_weights=None,
        **kwargs,
    ):
        super(PatchClassifier_pl, self).__init__()

        self.save_hyperparameters()

        args = inspect.signature(self.__init__).parameters
        for arg in args:
            if arg != "self":
                setattr(self, arg, locals()[arg])
        for key, value in {**locals(), **kwargs}.items():
            if key != "self":
                setattr(self, key, value)

        self.data_name = data_name

        self.path_classifier = PatchClassifier(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            encoder=encoder,
            from_file=from_file,
            dropprob=dropprob,
            dim_in=dim_in,
        )

        if data_weights is not None:
            data_weights = [1] * len(l_fns)
        self.data_weights = data_weights

        if not isinstance(l_fns, list):
            l_fns = [l_fns]
        self.loss = []
        self.task = []

        for i, l_fn in enumerate(l_fns):
            if l_fn == "cross_entropy":
                self.l = nn.CrossEntropyLoss(self.weights)

                def cross_entropy(x, y):
                    return self.l(x, y)  # * 10

                self.loss.append(cross_entropy)
                self.task.append("classification")

            if l_fn == "none":
                self.l = nn.CrossEntropyLoss(self.weights)

                def cross_entropy(x, y):
                    return self.l(x, y) * 0

                self.loss.append(cross_entropy)
                self.task.append("classification")

            elif l_fn == "focal":

                def focal_loss(x, y):
                    y_onehot = torch.nn.functional.one_hot(y, num_classes[i]).float()
                    return (
                        torchvision.ops.sigmoid_focal_loss(
                            x, y_onehot, reduction="mean"
                        )
                        * 20
                    )

                self.loss.append(focal_loss)
                self.task.append("classification")

            elif l_fn == "l1":

                def l1_loss(x, y):
                    x = torch.sigmoid(x)
                    x = (x - 0.5) * 10
                    y = (y - 0.5) * 10
                    return nn.L1Loss()(x, y)

                self.loss.append(l1_loss)
                self.task.append("regression")

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.len_epoch = len_epoch
        self.accumulate_grad_batches = accumulate_grad_batches
        self.weight_decay = weight_decay

        if class_names is not None:
            self.class_names = [
                [class_names_i[str(i)] for i in range(num_class)]
                for class_names_i, num_class in zip(class_names, num_classes)
            ]
        self.confusion_matrix = [
            np.zeros((num_class, num_class), dtype=int) for num_class in num_classes
        ]

        self.transform_train = DataAugmentation("train", apply_color_jitter=jitter)
        self.transform_val = DataAugmentation("val", apply_color_jitter=False)

    def forward(self, x, *args, **kwargs):
        x = self.path_classifier(x, *args, **kwargs)
        return x

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y, z = batch
        if self.trainer.training:
            x = self.transform_train(x)  # => we perform GPU/Batched data augmentation
        else:
            x = self.transform_val(x)
        x[:, -1] = (x[:, -1] > 0.1).to(x.dtype)
        return x, y, z

    def training_step(self, batch):
        images, labels, idx = batch
        opt = self.optimizers()
        self.log("lr", opt.param_groups[0]["lr"], sync_dist=True)

        assert not torch.isnan(images).any(), pdb.set_trace()

        assert images.mean() >= 0 and images.mean() <= 1
        assert images.std() >= 0 and images.std() <= 1

        self.training_mean = images.mean((0, 2, 3))
        self.training_std = images.std((0, 2, 3))

        outputs = self(images)
        loss = 0

        ids, counts = torch.unique(idx, return_counts=True)
        all_counts = sum(counts)  # should be batch size

        for id, count in zip(ids, counts):
            id_weight = (count / all_counts) * self.data_weights[id]
            if id == 0:
                start_idx = 0
            else:
                start_idx = np.cumsum(self.num_classes)[id - 1]

            end_idx = np.cumsum(self.num_classes)[id]
            outputs_tmp = outputs[idx == id][:, start_idx:end_idx]

            labels_tmp = torch.stack(
                [
                    labels[i]
                    for i in torch.arange(len(idx), device=labels[0].device)[idx == id]
                ]
            )
            target_tmp = labels_tmp.squeeze(1)
            loss += self.loss[id](outputs_tmp, target_tmp) * id_weight

        self.log("cuda utilization", get_gpu_utilization()[0])
        self.log(
            "cuda memory utilization",
            torch.cuda.memory_allocated(0)
            / torch.cuda.get_device_properties(0).total_memory,
        )

        import psutil

        self.log("cpu utilization", psutil.cpu_percent(interval=0))

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch):
        images, labels, idx = batch

        # from instanseg.utils.utils import display_as_grid
        # display_as_grid(images,ncols = 10)
        # plt.show()

        outputs = self(images)

        loss = 0

        ids, counts = torch.unique(idx, return_counts=True)
        all_counts = sum(counts)  # should be batch size

        for id, count in zip(ids, counts):
            id_weight = (count / all_counts) * self.data_weights[id]
            if id == 0:
                start_idx = 0
            else:
                start_idx = np.cumsum(self.num_classes)[id - 1]
            end_idx = np.cumsum(self.num_classes)[id]
            outputs_tmp = outputs[idx == id][:, start_idx:end_idx]

            labels_tmp = torch.stack(
                [
                    labels[i]
                    for i in torch.arange(len(idx), device=labels[0].device)[idx == id]
                ]
            )
            target_tmp = labels_tmp.squeeze(1)
            loss += self.loss[id](outputs_tmp, target_tmp) * id_weight

            # pdb.set_trace()

            if self.task[id] == "classification":
                preds = torch.argmax(outputs_tmp, dim=1)
                f1 = f1_score(
                    target_tmp.cpu().numpy(), preds.cpu().numpy(), average="macro"
                )
                self.log(f"val_f1_{self.data_name[id]}", f1, sync_dist=True)
                cm = confusion_matrix(
                    target_tmp.cpu().numpy(),
                    preds.cpu().numpy(),
                    labels=np.arange(self.num_classes[id]),
                )
                self.confusion_matrix[id] += cm

        self.log("val_loss", loss, sync_dist=True)

    def on_validation_epoch_end(self):
        for id in range(len(self.task)):
            if self.task[id] == "classification":
                # Plot the accumulated confusion matrix
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=self.confusion_matrix[id],
                    display_labels=self.class_names[id],
                )

                fig, ax = plt.subplots(figsize=(10, 10))
                disp.plot(ax=ax)
                plt.xticks(rotation=45)

                # Save the plot to a buffer
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)

                # Convert the buffer to a tensor
                img_tensor = ToTensor()(plt.imread(buf))

                # Log the image tensor
                self.logger.experiment.add_image(
                    f"confusion_matrix_{self.data_name[id]}",
                    img_tensor,
                    self.current_epoch,
                )

                # Clear the plot to free memory
                plt.close(fig)

                # Reset the confusion matrix for the next epoch
                self.confusion_matrix[id] = np.zeros(
                    (self.num_classes[id], self.num_classes[id]), dtype=int
                )

    # def configure_optimizers(self, weight_decay = 0): #1e-4

    #     num_epochs = self.num_epochs
    #     num_batches_per_epoch = self.len_epoch // self.batch_size
    #     T_max = (num_epochs * num_batches_per_epoch) // self.accumulate_grad_batches

    #     lr = 1e-3 * (128 / self.batch_size)**0.5
    #    # optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay = weight_decay)
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay = 0.1, eps = 1e-7)

    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)
    #     return {
    #             "optimizer": optimizer,
    #             "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
    #         }
    def configure_optimizers(self):
        num_epochs = self.num_epochs
        T_0 = int(((self.num_epochs * self.len_epoch) // self.batch_size) / 5)
        lr = 1e-3 * (128 / (self.batch_size * self.accumulate_grad_batches)) ** 0.5
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=self.weight_decay
        )

        #   scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0)

        return {
            "optimizer": optimizer,
            # "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class PatchDataset(Dataset):
    def __init__(
        self,
        data_file,
        split,
        chunk_size=1,
        batch_size=128,
        requested_pixel_size=None,
        data_pixel_size=None,
        in_memory=True,
        num_datasets=1,
        dataset_id=0,
    ):
        self.data_file = data_file
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.split = split
        self.load_data(in_memory=in_memory)
        from torchvision import transforms

        self.requested_pixel_size = requested_pixel_size
        self.data_pixel_size = data_pixel_size
        self.num_datasets = num_datasets
        self.dataset_id = dataset_id

    def load_data(self, in_memory=True):
        self.in_memory = in_memory

        if not in_memory:
            with h5py.File(self.data_file, "r") as f:
                self.num_samples = f[f"{self.split}/data"].shape[0]
                self.chunk_size = f[f"{self.split}/data"].chunks
                print(
                    f"{self.split} - chunk size",
                    self.chunk_size,
                    "num samples",
                    self.num_samples,
                )
                self.labels = torch.tensor(f[f"{self.split}/labels"][:])

            self.data = None
        else:
            with h5py.File(self.data_file, "r") as f:
                self.data = f[f"{self.split}/data"][:]
                self.labels = torch.tensor(f[f"{self.split}/labels"][:])
                self.num_samples = self.data.shape[0]
                print(f"{self.split} - loaded {self.num_samples} samples into memory")

        if not self.labels.is_floating_point():
            self.labels = self.labels.long()
            print(torch.unique(self.labels, return_counts=True))
        else:
            self.labels = torch.clamp(self.labels, 0, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.in_memory:
            image = torch.tensor(self.data[idx], dtype=torch.float32) / 255.0
            label = self.labels[idx]

        else:
            if self.data is None:
                self.data = h5py.File(self.data_file, "r")[f"{self.split}/data"]

            image = torch.tensor((self.data[idx])).float() / 255.0
            label = self.labels[idx]

        if self.requested_pixel_size is not None and self.data_pixel_size is not None:
            if not np.allclose(self.requested_pixel_size, self.data_pixel_size):
                scale = self.data_pixel_size / self.requested_pixel_size
                image = torch.nn.functional.interpolate(
                    image.unsqueeze(0), scale_factor=scale, mode="bilinear"
                ).squeeze(0)

        image[-1] = image[-1] / image[-1].max()

        return image, label, self.dataset_id


def get_sampling_weights(labels, weight=True):
    if not weight:
        return torch.ones_like(labels).float() / len(labels)
    assert len(torch.unique(labels)) == max(labels) + 1
    class_counts = torch.unique(labels, return_counts=True)[1]
    weights = 1 / ((class_counts.float() / class_counts.sum()) + 0.0001)
    weights = weights / weights.sum()
    sample_weights = torch.tensor([weights[int(i)] for i in labels]).unsqueeze(1)

    sample_weights = sample_weights / len(sample_weights)

    return sample_weights


class PatchDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_file,
        batch_size=128,
        len_epoch=100000,
        equal_sampling=False,
        num_workers=8,
        **kwargs,
    ):
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.equal_sampling = equal_sampling
        self.len_epoch = len_epoch
        self.num_workers = num_workers

    def setup(self, stage=None):
        if isinstance(self.data_file, list) and len(self.data_file) > 1:
            num_datasets = len(self.data_file)

            data_list = [
                PatchDataset(
                    data_file,
                    "train",
                    batch_size=self.batch_size,
                    num_datasets=num_datasets,
                    dataset_id=i,
                    **self.kwargs,
                )
                for i, data_file in enumerate(self.data_file)
            ]

            self.train_dataset = ConcatDataset(data_list)

            print(len(data_list), self.equal_sampling)

            self.train_dataset.weights = torch.cat(
                [
                    get_sampling_weights(d.labels, weight=self.equal_sampling[i])
                    for i, d in enumerate(data_list)
                ]
            )

            self.val_dataset = ConcatDataset(
                [
                    PatchDataset(
                        data_file,
                        "val",
                        batch_size=self.batch_size,
                        num_datasets=num_datasets,
                        dataset_id=i,
                        **self.kwargs,
                    )
                    for i, data_file in enumerate(self.data_file)
                ]
            )

        else:
            if isinstance(self.data_file, list):
                self.data_file = self.data_file[0]
            self.train_dataset = PatchDataset(
                self.data_file, "train", batch_size=self.batch_size, **self.kwargs
            )
            self.val_dataset = PatchDataset(
                self.data_file, "val", batch_size=self.batch_size, **self.kwargs
            )
            # self.test_dataset = PatchDataset(self.data_file, 'test', batch_size=self.batch_size, **kwargs)

    def train_dataloader(self):
        if True in self.equal_sampling:
            sample_weights = self.train_dataset.weights
            from torch.utils.data import WeightedRandomSampler

            sampler = WeightedRandomSampler(
                sample_weights.squeeze(1), num_samples=self.len_epoch, replacement=True
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sampler=sampler,
            )

        else:
            if torch.distributed.is_initialized():
                print("Using distributed sampler")
                sampler = torch.utils.data.distributed.DistributedSampler(
                    self.train_dataset
                )
            else:
                sampler = torch.utils.data.RandomSampler(
                    self.train_dataset, replacement=True, num_samples=self.len_epoch
                )

            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sampler=sampler,
                collate_fn=my_collate_fn,
            )

    def val_dataloader(self):
        # sampler = torch.utils.data.RandomSampler(self.val_dataset, replacement=True, num_samples=int(self.len_epoch ))
        # return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler,collate_fn = my_collate_fn)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=my_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


def read_dataset_attributes(data_file, weight: bool = False):
    with h5py.File(data_file, "r") as f:
        image_size = f[f"train/data"].shape[3]
        class_names = eval(f.attrs["class_names"])
        pixel_size = f.attrs["pixel_size"]
        num_classes = len(class_names)
        #   pdb.set_trace()
        if "dim_in" in f.attrs:
            dim_in = f.attrs["dim_in"]
        else:
            dim_in = 4

        if weight:
            labels = torch.tensor(f[f"train/labels"][:])

            if labels.shape[1] == 1:
                class_counts = torch.unique(labels, return_counts=True)[1]
            else:
                import rapids_singlecell as rsc
                import scanpy as sc

                adata = sc.AnnData(labels.numpy())
                rsc.get.anndata_to_GPU(adata)
                rsc.pp.scale(adata)
                rsc.pp.neighbors(adata, n_neighbors=5)
                rsc.tl.umap(adata)
                rsc.tl.leiden(adata, resolution=0.15, key_added="leiden")
                pdb.set_trace()
                y_pred = torch.tensor(adata.obs["leiden"].values.astype(np.int32))
                class_counts = torch.unique(y_pred, return_counts=True)[1]

            weights = 1 / ((class_counts.float() / class_counts.sum()) + 0.1)
            weights = weights / weights.sum()

        else:
            weights = None

    return {
        "class_names": class_names,
        "pixel_size": pixel_size,
        "num_classes": num_classes,
        "dim_in": dim_in,
        "image_size": image_size,
        "weights": weights,
    }


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)

    # import matplotlib
    # matplotlib.use("Agg")
    # parse arguments

    parser = argparse.ArgumentParser(description="Train a patch classifier")
    parser.add_argument(
        "--loss", type=str, default="cross_entropy", help="Loss function"
    )
    parser.add_argument("--encoder", type=str, default="mobilenet_v3", help="Model")
    parser.add_argument(
        "--data", type=str, default="pannuke", help="Dataset to train on"
    )
    parser.add_argument(
        "--pixel_size", type=float, default=0.5, help="Pixel size to run the model"
    )
    parser.add_argument("--output_dir", type=str, default="../outputs")
    parser.add_argument("--data_dir", type=str, default="../datasets")
    parser.add_argument("-e_s", "--experiment_str", type=str, default="0")
    parser.add_argument("-from_checkpoint", "--from_checkpoint", default=None, type=str)
    parser.add_argument("-e", "--num_epochs", type=int, default=250)
    parser.add_argument("-bs", "--batch_size", type=int, default=128)
    parser.add_argument("-nw", "--num_workers", type=int, default=6)
    parser.add_argument("-agb", "--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("-dp", "--dropprob", type=int, default=0.1)
    parser.add_argument("-wd", "--weight_decay", type=int, default=1e-4)
    parser.add_argument("-seed", "--random_seed", type=float, default=None)
    parser.add_argument(
        "-dw",
        "--data_weights",
        type=str,
        default=None,
        help="Weight to give to the loss of each dataset",
    )
    parser.add_argument(
        "-equal",
        "--use_equal",
        default=False,
        type=str,
        help="Whether to use equal sampling",
    )
    parser.add_argument(
        "-weight",
        "--weight",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
        help="Whether to weigth cross entropy",
    )
    parser.add_argument(
        "-jitter", "--jitter", default=False, type=lambda x: (str(x).lower() == "true")
    )
    parser.add_argument(
        "-in_mem",
        "--in_memory",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
    )

    args = parser.parse_args()
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    args.data = args.data.strip("[]").split(",")
    args.loss = args.loss.strip("[]").split(",")

    if args.data_weights is not None:
        args.data_weights = args.data_weights.strip("[]").split(",")
        assert len(args.data_weights) == len(args.data)

    if args.use_equal in ["True", "False"]:
        args.use_equal = [eval(args.use_equal)] * len(args.data)
    else:
        args.use_equal = args.use_equal.strip("[]").split(",")
        args.use_equal = [eval(i) for i in args.use_equal]

    assert len(args.data) == len(args.loss)
    assert len(args.data) == len(args.use_equal)

    in_memory = args.in_memory

    if not isinstance(args.data, list):
        args.data = [args.data]

    data_path = [f"{args.data_dir}/{data}.h5" for data in args.data]

    num_classes = []
    class_names = []
    for i in data_path:
        dataset_attributes = read_dataset_attributes(i, weight=args.weight)
        num_classes.append(dataset_attributes["num_classes"])
        class_names.append(dataset_attributes["class_names"])
        data_pixel_size = dataset_attributes["pixel_size"]
        dim_in = dataset_attributes["dim_in"]
        image_size = dataset_attributes["image_size"]
        weights = dataset_attributes["weights"]

    len_epoch = 100000

    data_module = PatchDataModule(
        data_path,
        equal_sampling=args.use_equal,
        requested_pixel_size=args.pixel_size,
        data_pixel_size=data_pixel_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        len_epoch=len_epoch,
        in_memory=args.in_memory,
    )

    patch_size = int(image_size / (args.pixel_size / data_pixel_size))

    accumulate_grad_batches = args.accumulate_grad_batches

    if args.from_checkpoint is not None:
        checkpoint_path = os.path.join(
            output_dir, "patch_classifier", args.from_checkpoint, "checkpoints"
        )
        last_checkpoint = sorted(os.listdir(checkpoint_path))[-1]
        print(f"Loading model from {last_checkpoint}")
        model = PatchClassifier_pl.load_from_checkpoint(
            checkpoint_path=os.path.join(checkpoint_path, last_checkpoint),
            strict=True,
            num_classes=num_classes,
            l_fns=args.loss,
            encoder=args.encoder,
            class_names=class_names,
            dropprob=args.dropprob,
            weight_decay=args.weight_decay,
            dim_in=dim_in,
            patch_size=patch_size,
            data_name=args.data,
            num_epochs=args.num_epochs,
            len_epoch=len_epoch,
            batch_size=args.batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            jitter=args.jitter,
            weights=weights,
            data_weights=args.data_weights,
        )

    else:
        model = PatchClassifier_pl(
            num_classes=num_classes,
            l_fns=args.loss,
            encoder=args.encoder,
            class_names=class_names,
            dropprob=args.dropprob,
            weight_decay=args.weight_decay,
            dim_in=dim_in,
            patch_size=patch_size,
            data_name=args.data,
            num_epochs=args.num_epochs,
            len_epoch=len_epoch,
            batch_size=args.batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            jitter=args.jitter,
            weights=weights,
            data_weights=args.data_weights,
        )

    tb_logger = pl.loggers.TensorBoardLogger(
        output_dir, name="patch_classifier", version=args.experiment_str
    )

    print("Run the command below to start tensorboard:")
    print(f"tensorboard --logdir {output_dir}/{tb_logger.name}")
    print("\n")

    progress_bar = TQDMProgressBar(refresh_rate=3)

    # torch.set_float32_matmul_precision('medium')

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        logger=tb_logger,
        precision="16-mixed",
        callbacks=[progress_bar],
        devices=-1,
        #        accumulate_grad_batches=accumulate_grad_batches,
        # strategy='ddp',
        profiler="simple",
    )

    trainer.fit(model, data_module)
