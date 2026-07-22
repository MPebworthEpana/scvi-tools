from unittest.mock import MagicMock, patch

import scvi
from scvi.data import synthetic_iid
from scvi.model import MULTIVI
from scvi.train import Trainer
from scvi.train._callbacks import LoudEarlyStopping


def test_trainer_keeps_early_stopping_under_ddp_strategy():
    captured = {}

    def fake_pl_trainer_init(self, **kwargs):
        captured.update(kwargs)
        self.callbacks = kwargs.get("callbacks", [])
        self.check_val_every_n_epoch = kwargs.get("check_val_every_n_epoch")

    with patch.object(Trainer.__bases__[0], "__init__", fake_pl_trainer_init):
        Trainer(
            early_stopping=True,
            strategy="ddp_find_unused_parameters_true",
            enable_progress_bar=False,
            logger=False,
        )

    assert any(isinstance(c, LoudEarlyStopping) for c in captured["callbacks"])
    assert captured["check_val_every_n_epoch"] == 1


def test_multivi_early_stopping_warmup_defaults_to_kl_warmup():
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
    model = MULTIVI(
        mdata,
        n_latent=5,
        n_genes=50,
        n_regions=50,
    )

    with patch.object(model, "_train_runner_cls") as mock_runner_cls:
        mock_runner_cls.return_value = MagicMock(return_value=None)
        model.train(
            max_epochs=1,
            early_stopping=True,
            n_epochs_kl_warmup=2,
        )
        trainer_kwargs = mock_runner_cls.call_args.kwargs
        assert trainer_kwargs["early_stopping_warmup_epochs"] == 2


def test_multivi_early_stopping_warmup_respects_explicit_override():
    mdata = synthetic_iid(return_mudata=True)
    MULTIVI.setup_mudata(
        mdata,
        batch_key="batch",
        modalities={
            "rna_layer": "rna",
            "protein_layer": "protein_expression",
            "atac_layer": "accessibility",
        },
    )
    model = MULTIVI(
        mdata,
        n_latent=5,
        n_genes=50,
        n_regions=50,
    )

    with patch.object(model, "_train_runner_cls") as mock_runner_cls:
        mock_runner_cls.return_value = MagicMock(return_value=None)
        model.train(
            max_epochs=1,
            early_stopping=True,
            n_epochs_kl_warmup=50,
            early_stopping_warmup_epochs=10,
        )
        trainer_kwargs = mock_runner_cls.call_args.kwargs
        assert trainer_kwargs["early_stopping_warmup_epochs"] == 10
