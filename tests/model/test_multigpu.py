import os
import shutil
import subprocess
from pathlib import Path

import anndata
import numpy as np
import pytest
import scipy.sparse as sp
import torch
import zarr
from anndata._core.sparse_dataset import BaseCompressedSparseDataset
from anndata.io import sparse_dataset
import mudata as md
from mudata import MuData

import scvi
from scvi.model import MULTIVI, PEAKVI, TOTALVI, CondSCVI, LinearSCVI
from scvi.train._callbacks import LoudEarlyStopping


def _make_trimodal_zarr_mudata(
    store_path: Path | str,
    *,
    batch_size: int = 64,
    n_genes: int = 50,
    n_regions: int = 50,
    n_adts: int = 20,
    n_batches: int = 2,
) -> tuple[MuData, MuData, Path]:
    """Build trimodal MuData with RNA/ATAC CSR zarr groups and dense ADT zarr.Array."""
    store_path = Path(store_path)
    if store_path.exists():
        shutil.rmtree(store_path)

    mdata_raw = scvi.data.synthetic_iid(
        return_mudata=True,
        batch_size=batch_size,
        n_genes=n_genes,
        n_proteins=n_adts,
        n_regions=n_regions,
        n_batches=n_batches,
    )
    n_obs = mdata_raw.n_obs
    mdata = MuData(
        {
            "RNA": mdata_raw.mod["rna"],
            "ATAC": mdata_raw.mod["accessibility"],
            "ADT": mdata_raw.mod["protein_expression"],
        }
    )
    mdata.obs = mdata_raw.obs.copy()
    mdata.mod["RNA"].X = sp.random(
        n_obs, n_genes, density=0.2, format="csr", dtype=np.float32
    )
    mdata.mod["ATAC"].X = sp.random(
        n_obs, n_regions, density=0.15, format="csr", dtype=np.float32
    )
    adt_x = mdata.mod["ADT"].X
    adt_dense = adt_x.toarray() if sp.issparse(adt_x) else np.asarray(adt_x)
    mdata.mod["ADT"].X = adt_dense.astype(np.float32, copy=False)

    previous = anndata.settings.allow_write_nullable_strings
    anndata.settings.allow_write_nullable_strings = True
    try:
        mdata.write_zarr(store_path)
    finally:
        anndata.settings.allow_write_nullable_strings = previous

    backed = md.read_zarr(store_path)
    root = zarr.open(str(store_path), mode="r")
    for mod_key in ("RNA", "ATAC"):
        x_group = root["mod"][mod_key]["X"]
        enc = x_group.attrs.get("encoding-type", "")
        if enc in ("csr_matrix", "csc_matrix"):
            backed.mod[mod_key].X = sparse_dataset(x_group)

    backed.mod["ADT"].X = zarr.open_array(str(store_path / "mod/ADT/X"), mode="r")
    return mdata, backed, store_path


def _assert_trimodal_zarr_backed(mdata_backed: MuData) -> None:
    assert isinstance(mdata_backed.mod["RNA"].X, BaseCompressedSparseDataset)
    assert isinstance(mdata_backed.mod["ATAC"].X, BaseCompressedSparseDataset)
    assert isinstance(mdata_backed.mod["ADT"].X, zarr.Array)


@pytest.mark.multigpu
@pytest.mark.parametrize("unlabeled_cat", ["label_0", "unknown"])
def test_scanvi_from_scvi_multigpu(unlabeled_cat: str):
    import scvi
    from scvi.model import SCVI

    adata = scvi.data.synthetic_iid()

    SCVI.setup_anndata(adata)

    datasplitter_kwargs = {}
    datasplitter_kwargs["drop_dataset_tail"] = True
    datasplitter_kwargs["drop_last"] = False

    model = SCVI(adata)

    print("multi GPU SCVI train")
    model.train(
        max_epochs=1,
        check_val_every_n_epoch=1,
        accelerator="gpu",
        devices=-1,
        datasplitter_kwargs=datasplitter_kwargs,
        strategy="ddp_find_unused_parameters_true",
    )
    print("done")
    assert len(model.history["elbo_train"]) == 1
    assert model.is_trained
    adata.obsm["scVI"] = model.get_latent_representation()

    datasplitter_kwargs = {}
    datasplitter_kwargs["drop_dataset_tail"] = True
    datasplitter_kwargs["drop_last"] = False

    print("multi GPU scanvi load from scvi model")
    model_scanvi = scvi.model.SCANVI.from_scvi_model(
        model,
        adata=adata,
        labels_key="labels",
        unlabeled_category=unlabeled_cat,
    )
    print("done")
    print("multi GPU scanvi train from scvi")
    model_scanvi.train(
        max_epochs=1,
        train_size=0.5,
        check_val_every_n_epoch=1,
        accelerator="gpu",
        devices=-1,
        strategy="ddp_find_unused_parameters_true",
        datasplitter_kwargs=datasplitter_kwargs,
    )
    print("done")
    adata.obsm["scANVI"] = model_scanvi.get_latent_representation()

    assert model_scanvi.is_trained


@pytest.mark.multigpu
@pytest.mark.parametrize("unlabeled_cat", ["label_0", "unknown"])
def test_scanvi_from_scratch_multigpu(unlabeled_cat: str):
    import scvi
    from scvi.model import SCANVI

    adata = scvi.data.synthetic_iid()

    SCANVI.setup_anndata(
        adata,
        labels_key="labels",
        unlabeled_category=unlabeled_cat,
        batch_key="batch",
    )

    datasplitter_kwargs = {}
    datasplitter_kwargs["drop_dataset_tail"] = True
    datasplitter_kwargs["drop_last"] = False

    model = SCANVI(adata, n_latent=10)

    print("multi GPU scanvi train from scratch")
    model.train(
        max_epochs=1,
        train_size=0.5,
        check_val_every_n_epoch=1,
        accelerator="gpu",
        devices=-1,
        datasplitter_kwargs=datasplitter_kwargs,
        strategy="ddp_find_unused_parameters_true",
    )
    print("done")
    assert len(model.history["elbo_train"]) == 1
    assert model.is_trained


@pytest.mark.multigpu
def test_totalvi_multigpu():
    adata = scvi.data.synthetic_iid()
    protein_adata = scvi.data.synthetic_iid(n_genes=50)
    mdata = MuData({"rna": adata, "protein": protein_adata})
    TOTALVI.setup_mudata(
        mdata,
        batch_key="batch",
        modalities={"rna_layer": "rna", "batch_key": "rna", "protein_layer": "protein"},
    )
    n_latent = 10
    model = TOTALVI(mdata, n_latent=n_latent)
    model.train(
        1,
        train_size=0.5,
        check_val_every_n_epoch=1,
        accelerator="gpu",
        devices=-1,
        strategy="ddp_find_unused_parameters_true",
    )
    assert len(model.history["elbo_train"]) == 1
    assert model.is_trained is True


@pytest.mark.multigpu
def test_multivi_multigpu():
    mdata = scvi.data.synthetic_iid(return_mudata=True)
    MULTIVI.setup_mudata(
        mdata,
        batch_key="batch",
        modalities={
            "rna_layer": "rna",
            "protein_layer": "protein_expression",
            "atac_layer": "accessibility",
        },
    )
    n_latent = 10
    model = MULTIVI(
        mdata,
        n_latent=n_latent,
        n_genes=50,
        n_regions=50,
    )
    model.train(
        1,
        train_size=0.5,
        check_val_every_n_epoch=1,
        accelerator="gpu",
        devices=-1,
        strategy="ddp_find_unused_parameters_true",
    )
    assert len(model.history["elbo_train"]) == 1
    assert model.is_trained is True


@pytest.mark.multigpu
def test_multivi_multigpu_ddp_safe_nan_skip_on_one_rank(save_path: str):
    """Inject NaN loss on rank 0 only; DDP training must not hang on skip."""
    import torch.distributed as dist

    from scvi.module.base import LossOutput
    from scvi.train import AdversarialTrainingPlan

    n_devices = torch.cuda.device_count()
    assert n_devices > 1, f"Need >1 GPU for multi-GPU test, got {n_devices}"

    zarr_store = Path(save_path) / "multivi_ddp_nan_skip.zarr"
    _, mdata_backed, _ = _make_trimodal_zarr_mudata(
        zarr_store,
        batch_size=64,
        n_genes=50,
        n_regions=50,
        n_adts=20,
    )
    _assert_trimodal_zarr_backed(mdata_backed)

    MULTIVI.setup_mudata(
        mdata_backed,
        batch_key="batch",
        modalities={
            "rna_layer": "RNA",
            "atac_layer": "ATAC",
            "protein_layer": "ADT",
        },
    )
    model = MULTIVI(
        mdata_backed,
        n_latent=10,
        n_genes=50,
        n_regions=50,
    )

    original_forward = AdversarialTrainingPlan.forward
    nan_injections_remaining = {"count": 8}

    def forward_with_rank0_nan(self, batch, *args, **kwargs):
        inference_outputs, generative_outputs, scvi_loss = original_forward(
            self, batch, *args, **kwargs
        )
        if (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_rank() == 0
            and nan_injections_remaining["count"] > 0
        ):
            nan_injections_remaining["count"] -= 1
            scvi_loss = LossOutput(
                loss=torch.tensor(float("nan"), device=scvi_loss.loss.device),
                n_obs_minibatch=scvi_loss.n_obs_minibatch,
            )
        return inference_outputs, generative_outputs, scvi_loss

    AdversarialTrainingPlan.forward = forward_with_rank0_nan
    try:
        model.train(
            max_epochs=1,
            train_size=0.5,
            check_val_every_n_epoch=1,
            accelerator="gpu",
            devices=-1,
            strategy="ddp_find_unused_parameters_true",
            batch_size=32,
            adversarial_mixing=True,
            datasplitter_kwargs={
                "drop_last": False,
                "block_size": 32,
                "shuffle_buffer_blocks": 1,
                "num_workers": 0,
            },
        )
    finally:
        AdversarialTrainingPlan.forward = original_forward

    assert model.is_trained is True
    assert model.trainer is not None
    assert model.trainer.world_size > 1
    assert nan_injections_remaining["count"] < 8

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.multigpu
def test_multivi_multigpu_early_stopping(save_path: str):
    """DDP MultiVI with early stopping should stop before max_epochs without hanging."""
    import torch.distributed as dist

    n_devices = torch.cuda.device_count()
    assert n_devices > 1, f"Need >1 GPU for multi-GPU test, got {n_devices}"

    zarr_store = Path(save_path) / "multivi_ddp_early_stop.zarr"
    _, mdata_backed, _ = _make_trimodal_zarr_mudata(
        zarr_store,
        batch_size=64,
        n_genes=50,
        n_regions=50,
        n_adts=20,
    )
    _assert_trimodal_zarr_backed(mdata_backed)

    MULTIVI.setup_mudata(
        mdata_backed,
        batch_key="batch",
        modalities={
            "rna_layer": "RNA",
            "atac_layer": "ATAC",
            "protein_layer": "ADT",
        },
    )
    model = MULTIVI(
        mdata_backed,
        n_latent=10,
        n_genes=50,
        n_regions=50,
    )

    max_epochs = 100
    try:
        model.train(
            max_epochs=max_epochs,
            train_size=0.5,
            lr=0,
            early_stopping=True,
            early_stopping_patience=5,
            early_stopping_warmup_epochs=0,
            check_val_every_n_epoch=1,
            accelerator="gpu",
            devices=-1,
            strategy="ddp_find_unused_parameters_true",
            batch_size=32,
            adversarial_mixing=True,
            datasplitter_kwargs={
                "drop_last": False,
                "block_size": 32,
                "shuffle_buffer_blocks": 1,
                "num_workers": 0,
            },
        )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    assert model.is_trained is True
    assert model.trainer is not None
    assert model.trainer.world_size > 1
    assert len(model.history["elbo_train"]) < max_epochs
    assert any(isinstance(c, LoudEarlyStopping) for c in model.trainer.callbacks)


@pytest.mark.multigpu
def test_multivi_multigpu_save_and_latent_use_rank0_device(save_path: str):
    """Multi-GPU DDP + zarr-backed MuData: save/inference use rank-0 device only.

    Uses lazy-loaded count matrices on disk:

    * RNA and ATAC: zarr CSR groups opened as ``CSRDataset``
    * ADT: dense ``zarr.Array`` at ``mod/ADT/X``

    Training streams through :class:`~scvi.dataloaders.ZarrMultiVIDataModule`.
    Post-training :meth:`~scvi.model.MULTIVI.save` and
    :meth:`~scvi.model.MULTIVI.get_latent_representation` must not fan out across
    all DDP ranks/devices.
    """
    from torch.nn.parallel import DistributedDataParallel

    n_devices = torch.cuda.device_count()
    assert n_devices > 1, f"Need >1 GPU for multi-GPU test, got {n_devices}"

    n_genes = 50
    n_regions = 50
    n_adts = 20
    n_latent = 10
    zarr_store = Path(save_path) / "multivi_trimodal.zarr"
    _, mdata_backed, zarr_store = _make_trimodal_zarr_mudata(
        zarr_store,
        batch_size=64,
        n_genes=n_genes,
        n_regions=n_regions,
        n_adts=n_adts,
    )
    _assert_trimodal_zarr_backed(mdata_backed)
    n_obs = mdata_backed.n_obs

    MULTIVI.setup_mudata(
        mdata_backed,
        batch_key="batch",
        modalities={
            "rna_layer": "RNA",
            "atac_layer": "ATAC",
            "protein_layer": "ADT",
        },
    )
    model = MULTIVI(
        mdata_backed,
        n_latent=n_latent,
        n_genes=n_genes,
        n_regions=n_regions,
    )

    model.train(
        max_epochs=1,
        train_size=0.5,
        check_val_every_n_epoch=1,
        accelerator="gpu",
        devices=-1,
        strategy="ddp_find_unused_parameters_true",
        batch_size=32,
        datasplitter_kwargs={
            "drop_last": False,
            "block_size": 32,
            "shuffle_buffer_blocks": 1,
            "num_workers": 0,
        },
    )

    assert model.is_trained is True
    assert model.trainer is not None
    assert model.trainer.is_global_zero
    assert model.trainer.world_size > 1
    assert model._zarr_dataloader_kwargs is not None

    # DDP leaves the process group alive; ZarrDataset._rank_indices() would then
    # return only this rank's slice during inference. Tear down before save/latent.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    # Post-training model state should live on a single GPU (rank 0), not all devices.
    module_devices = {p.device for p in model.module.parameters()}
    assert len(module_devices) == 1
    device = next(iter(module_devices))
    assert device.type == "cuda"
    assert device.index == 0
    assert model.device.index == 0
    assert not isinstance(model.module, DistributedDataParallel)

    inference_devices: list[int | None] = []
    original_inference = model.module.inference

    def _track_inference(*args, **kwargs):
        if torch.cuda.is_available():
            inference_devices.append(torch.cuda.current_device())
        return original_inference(*args, **kwargs)

    model.module.inference = _track_inference

    latent = model.get_latent_representation()
    assert latent.shape == (n_obs, n_latent)
    assert inference_devices, "Expected at least one inference call during latent extraction"
    assert all(dev == 0 for dev in inference_devices)

    model_dir = os.path.join(save_path, "multivi_multigpu_rank0")
    save_calls = {"count": 0}
    original_torch_save = torch.save

    def _counting_torch_save(*args, **kwargs):
        save_calls["count"] += 1
        return original_torch_save(*args, **kwargs)

    torch.save = _counting_torch_save
    try:
        model.save(model_dir, overwrite=True)
    finally:
        torch.save = original_torch_save

    assert save_calls["count"] == 1
    assert os.path.isdir(model_dir)
    assert os.path.isfile(os.path.join(model_dir, "model.pt"))

    loaded = MULTIVI.load(model_dir, adata=mdata_backed)
    assert loaded.device.index == 0
    loaded_latent = loaded.get_latent_representation(adata=mdata_backed)
    assert loaded_latent.shape == (n_obs, n_latent)


@pytest.mark.multigpu
def test_peakvi_multigpu():
    adata = scvi.data.synthetic_iid()
    PEAKVI.setup_anndata(
        adata,
        batch_key="batch",
    )

    model = PEAKVI(
        adata,
        model_depth=False,
    )

    model.train(
        max_epochs=1,
        train_size=0.5,
        check_val_every_n_epoch=1,
        accelerator="gpu",
        devices=-1,
        strategy="ddp_find_unused_parameters_true",
    )
    assert len(model.history["elbo_train"]) == 1
    assert model.is_trained


@pytest.mark.multigpu
def test_condscvi_multigpu():
    adata = scvi.data.synthetic_iid()
    adata.obs["overclustering_vamp"] = list(range(adata.n_obs))
    CondSCVI.setup_anndata(
        adata,
        labels_key="labels",
    )

    model = CondSCVI(adata)

    model.train(
        max_epochs=1,
        train_size=0.9,
        check_val_every_n_epoch=1,
        accelerator="gpu",
        devices=-1,
        strategy="ddp_find_unused_parameters_true",
    )
    assert len(model.history["elbo_train"]) == 1
    assert model.is_trained


@pytest.mark.multigpu
def test_linearcvi_multigpu():
    adata = scvi.data.synthetic_iid()
    adata = adata[:, :10].copy()
    LinearSCVI.setup_anndata(adata)
    model = LinearSCVI(adata, n_latent=10)

    model.train(
        max_epochs=1,
        train_size=0.5,
        check_val_every_n_epoch=1,
        accelerator="gpu",
        devices=-1,
        strategy="ddp_find_unused_parameters_true",
    )
    assert len(model.history["elbo_train"]) == 1
    assert model.is_trained


@pytest.mark.multigpu
def test_scvi_train_ddp(save_path: str):
    training_code = """
import torch
import scvi
from scvi.model import SCVI

adata = scvi.data.synthetic_iid()
SCVI.setup_anndata(adata)

model = SCVI(adata)

model.train(
    max_epochs=1,
    check_val_every_n_epoch=1,
    accelerator="gpu",
    devices=-1,
    strategy="ddp_find_unused_parameters_true",
)
assert model.is_trained
"""
    # Define the file path for the temporary script in the current working directory
    temp_file_path = os.path.join(save_path, "train_scvi_ddp_temp.py")

    # Write the training code to the file in the current working directory
    with open(temp_file_path, "w") as temp_file:
        temp_file.write(training_code)
        print(f"Temporary Python file created at: {temp_file_path}")

    def launch_ddp(world_size, temp_file_path):
        # Command to run the script via torchrun
        command = [
            "torchrun",
            "--nproc_per_node=" + str(world_size),  # Specify the number of GPUs
            temp_file_path,  # Your original script
        ]
        # Use subprocess to run the command
        try:
            # Run the command, wait for it to finish & clean up the temporary file
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            os.remove(temp_file_path)
            print(f"Error occurred while running the DDP training: {e}")
            raise
        finally:
            os.remove(temp_file_path)

    launch_ddp(torch.cuda.device_count(), temp_file_path)


@pytest.mark.multigpu
@pytest.mark.parametrize("unlabeled_cat", ["label_0", "unknown"])
def test_scanvi_train_ddp(unlabeled_cat: str, save_path: str):
    training_code = """
import torch
import scvi
from scvi.model import SCANVI

adata = scvi.data.synthetic_iid()
SCANVI.setup_anndata(
    adata,
    "labels",
    unlabeled_cat,
    batch_key="batch",
)

model = SCANVI(adata, n_latent=10)

datasplitter_kwargs = {}
datasplitter_kwargs["drop_dataset_tail"] = True
datasplitter_kwargs["drop_last"] = False

model.train(
    max_epochs=1,
    train_size=0.5,
    check_val_every_n_epoch=1,
    accelerator="gpu",
    devices=-1,
    strategy="ddp_find_unused_parameters_true",
    datasplitter_kwargs=datasplitter_kwargs,
)

assert model.is_trained
"""
    # Define the file path for the temporary script in the current working directory
    temp_file_path = os.path.join(save_path, "train_scanvi_ddp_temp.py")

    # Write the training code to the file in the current working directory
    with open(temp_file_path, "w") as temp_file:
        temp_file.write(training_code)
        print(f"Temporary Python file created at: {temp_file_path}")

    def launch_ddp(world_size, temp_file_path):
        # Command to run the script via torchrun
        command = [
            "torchrun",
            "--nproc_per_node=" + str(world_size),  # Specify the number of GPUs
            temp_file_path,  # Your original script
        ]
        # Use subprocess to run the command
        try:
            # Run the command, wait for it to finish & clean up the temporary file
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            os.remove(temp_file_path)
            print(f"Error occurred while running the DDP training: {e}")
            raise
        finally:
            os.remove(temp_file_path)

    launch_ddp(torch.cuda.device_count(), temp_file_path)
