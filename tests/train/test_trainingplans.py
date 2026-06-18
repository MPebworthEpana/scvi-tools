import pytest
import torch
from unittest.mock import MagicMock, patch

import scvi
from scvi import REGISTRY_KEYS
from scvi.data import synthetic_iid
from scvi.model import SCVI
from scvi.module.base import LossOutput
from scvi.train import AdversarialTrainingPlan, TrainingPlan
from scvi.train._constants import METRIC_KEYS
from scvi.train._trainingplans import (
    SemiSupervisedAdversarialTrainingPlan,
    _compute_kl_weight,
)


def _mock_trainer(plan):
    plan.trainer = MagicMock()
    plan.trainer.logger = MagicMock()
    plan.trainer.current_epoch = 0
    plan.trainer.global_step = 0
    plan.trainer.is_global_zero = True
    plan.log = MagicMock()
    plan.compute_and_log_metrics = MagicMock()
    plan.manual_backward = MagicMock()
    plan.on_step = False
    plan.on_epoch = True


def _assert_skip_warning(warn_mock, reason: str, streak: int):
    assert warn_mock.call_count >= 1
    message = warn_mock.call_args[0][0]
    assert reason in message
    assert f"consecutive_skips={streak}" in message


def _finite_forward_return(device, n_latent):
    z = torch.zeros(4, n_latent, device=device)
    loss = torch.tensor(1.0, device=device)
    inference_outputs = {"z": z}
    scvi_loss = LossOutput(loss=loss, n_obs_minibatch=4)
    return inference_outputs, None, scvi_loss


def _make_adversarial_plan(adversarial_classifier=False, **plan_kwargs):
    adata = synthetic_iid()
    SCVI.setup_anndata(adata, batch_key="batch")
    vae = SCVI(adata)
    plan = AdversarialTrainingPlan(
        vae.module,
        adversarial_classifier=adversarial_classifier,
        **plan_kwargs,
    )
    _mock_trainer(plan)
    opt1 = MagicMock()
    batch = {REGISTRY_KEYS.BATCH_KEY: torch.zeros(4, 1, dtype=torch.long)}
    if plan.adversarial_classifier is not False:
        opt2 = MagicMock()
        plan.optimizers = MagicMock(return_value=[opt1, opt2])
        return plan, opt1, opt2, batch
    plan.optimizers = MagicMock(return_value=opt1)
    return plan, opt1, None, batch


def _make_semisupervised_adversarial_plan(adversarial_classifier=False, **plan_kwargs):
    adata = synthetic_iid(n_labels=3)
    SCVI.setup_anndata(adata, batch_key="batch")
    vae = SCVI(adata)
    plan = SemiSupervisedAdversarialTrainingPlan(
        vae.module,
        n_classes=3,
        adversarial_classifier=adversarial_classifier,
        **plan_kwargs,
    )
    _mock_trainer(plan)
    opt1 = MagicMock()
    batch = {REGISTRY_KEYS.BATCH_KEY: torch.zeros(4, 1, dtype=torch.long)}
    if plan.adversarial_classifier is not False:
        opt2 = MagicMock()
        plan.optimizers = MagicMock(return_value=[opt1, opt2])
        return plan, opt1, opt2, batch
    plan.optimizers = MagicMock(return_value=opt1)
    return plan, opt1, None, batch


@pytest.mark.parametrize("adversarial_classifier", [False, True])
def test_adversarial_plan_warns_on_nonfinite_loss_skip(adversarial_classifier):
    plan, opt1, opt2, batch = _make_adversarial_plan(
        adversarial_classifier=adversarial_classifier
    )
    device = batch[REGISTRY_KEYS.BATCH_KEY].device

    def forward_with_nan(*args, **kwargs):
        inference_outputs, _, _ = _finite_forward_return(device, plan.module.n_latent)
        scvi_loss = LossOutput(
            loss=torch.tensor(float("nan"), device=device),
            n_obs_minibatch=4,
        )
        return inference_outputs, None, scvi_loss

    plan.forward = forward_with_nan
    with patch("scvi.train._trainingplans.warnings.warn") as warn_mock:
        plan.training_step(batch, 0)

    _assert_skip_warning(warn_mock, "non-finite main loss", 1)
    assert plan._main_skip_streak == 1
    assert plan._main_skip_total == 1


@pytest.mark.parametrize("adversarial_classifier", [False, True])
def test_adversarial_plan_warns_on_nonfinite_grad_skip(adversarial_classifier):
    plan, opt1, opt2, batch = _make_adversarial_plan(
        adversarial_classifier=adversarial_classifier
    )
    device = batch[REGISTRY_KEYS.BATCH_KEY].device
    plan.forward = lambda *args, **kwargs: _finite_forward_return(
        device, plan.module.n_latent
    )

    with (
        patch(
            "scvi.train._trainingplans.torch.nn.utils.clip_grad_norm_",
            side_effect=RuntimeError("nonfinite grad"),
        ),
        patch("scvi.train._trainingplans.warnings.warn") as warn_mock,
    ):
        plan.training_step(batch, 0)

    _assert_skip_warning(warn_mock, "non-finite main gradients", 1)
    assert plan._main_skip_streak == 1


@pytest.mark.parametrize("adversarial_classifier", [False, True])
def test_adversarial_plan_repeated_main_skip_increments_streak(adversarial_classifier):
    plan, opt1, opt2, batch = _make_adversarial_plan(
        adversarial_classifier=adversarial_classifier
    )
    device = batch[REGISTRY_KEYS.BATCH_KEY].device

    def forward_with_nan(*args, **kwargs):
        inference_outputs, _, _ = _finite_forward_return(device, plan.module.n_latent)
        scvi_loss = LossOutput(
            loss=torch.tensor(float("nan"), device=device),
            n_obs_minibatch=4,
        )
        return inference_outputs, None, scvi_loss

    plan.forward = forward_with_nan
    with patch("scvi.train._trainingplans.warnings.warn") as warn_mock:
        plan.training_step(batch, 0)
        plan.training_step(batch, 1)

    assert plan._main_skip_streak == 2
    assert plan._main_skip_total == 2
    _assert_skip_warning(warn_mock, "non-finite main loss", 2)


def test_adversarial_plan_main_skip_streak_resets_on_success():
    plan, opt1, opt2, batch = _make_adversarial_plan(adversarial_classifier=False)
    device = batch[REGISTRY_KEYS.BATCH_KEY].device

    def forward_with_nan(*args, **kwargs):
        inference_outputs, _, _ = _finite_forward_return(device, plan.module.n_latent)
        scvi_loss = LossOutput(
            loss=torch.tensor(float("nan"), device=device),
            n_obs_minibatch=4,
        )
        return inference_outputs, None, scvi_loss

    good_forward = lambda *args, **kwargs: _finite_forward_return(
        device, plan.module.n_latent
    )

    plan.forward = forward_with_nan
    with patch("scvi.train._trainingplans.warnings.warn"):
        plan.training_step(batch, 0)
    assert plan._main_skip_streak == 1

    plan.forward = good_forward
    with (
        patch(
            "scvi.train._trainingplans.torch.nn.utils.clip_grad_norm_",
            return_value=1.0,
        ),
        patch("scvi.train._trainingplans.warnings.warn"),
    ):
        plan.training_step(batch, 1)

    assert plan._main_skip_streak == 0


def test_adversarial_plan_warns_on_classifier_skip():
    plan, opt1, opt2, batch = _make_adversarial_plan(adversarial_classifier=True)
    device = batch[REGISTRY_KEYS.BATCH_KEY].device
    plan.forward = lambda *args, **kwargs: _finite_forward_return(
        device, plan.module.n_latent
    )

    with (
        patch(
            "scvi.train._trainingplans.torch.nn.utils.clip_grad_norm_",
            side_effect=[1.0, RuntimeError("nonfinite grad")],
        ),
        patch("scvi.train._trainingplans.warnings.warn") as warn_mock,
    ):
        plan.training_step(batch, 0)

    _assert_skip_warning(warn_mock, "non-finite classifier gradients", 1)
    assert plan._cls_skip_streak == 1
    opt1.step.assert_called_once()
    opt2.step.assert_not_called()


@pytest.mark.parametrize("adversarial_classifier", [False, True])
def test_semisupervised_adversarial_plan_warns_on_nonfinite_loss_skip(adversarial_classifier):
    plan, opt1, opt2, batch = _make_semisupervised_adversarial_plan(
        adversarial_classifier=adversarial_classifier
    )
    device = batch[REGISTRY_KEYS.BATCH_KEY].device

    def forward_with_nan(*args, **kwargs):
        inference_outputs, _, _ = _finite_forward_return(device, plan.module.n_latent)
        scvi_loss = LossOutput(
            loss=torch.tensor(float("nan"), device=device),
            n_obs_minibatch=4,
        )
        return inference_outputs, None, scvi_loss

    plan.forward = forward_with_nan
    with patch("scvi.train._trainingplans.warnings.warn") as warn_mock:
        plan.training_step(batch, 0)

    _assert_skip_warning(warn_mock, "non-finite main loss", 1)
    assert plan._main_skip_streak == 1


@pytest.mark.parametrize("adversarial_classifier", [False, True])
def test_semisupervised_adversarial_plan_repeated_main_skip_increments_streak(
    adversarial_classifier,
):
    plan, opt1, opt2, batch = _make_semisupervised_adversarial_plan(
        adversarial_classifier=adversarial_classifier
    )
    device = batch[REGISTRY_KEYS.BATCH_KEY].device

    def forward_with_nan(*args, **kwargs):
        inference_outputs, _, _ = _finite_forward_return(device, plan.module.n_latent)
        scvi_loss = LossOutput(
            loss=torch.tensor(float("nan"), device=device),
            n_obs_minibatch=4,
        )
        return inference_outputs, None, scvi_loss

    plan.forward = forward_with_nan
    with patch("scvi.train._trainingplans.warnings.warn") as warn_mock:
        plan.training_step(batch, 0)
        plan.training_step(batch, 1)

    assert plan._main_skip_streak == 2
    _assert_skip_warning(warn_mock, "non-finite main loss", 2)


@pytest.mark.parametrize("adversarial_classifier", [False, True])
def test_adversarial_plan_skips_nonfinite_loss(adversarial_classifier):
    plan, opt1, opt2, batch = _make_adversarial_plan(
        adversarial_classifier=adversarial_classifier
    )
    device = batch[REGISTRY_KEYS.BATCH_KEY].device

    def forward_with_nan(*args, **kwargs):
        inference_outputs, _, _ = _finite_forward_return(device, plan.module.n_latent)
        scvi_loss = LossOutput(
            loss=torch.tensor(float("nan"), device=device),
            n_obs_minibatch=4,
        )
        return inference_outputs, None, scvi_loss

    plan.forward = forward_with_nan
    result = plan.training_step(batch, 0)

    assert torch.isfinite(result).all()
    opt1.step.assert_not_called()
    if opt2 is not None:
        opt2.step.assert_not_called()


@pytest.mark.parametrize("adversarial_classifier", [False, True])
def test_adversarial_plan_skips_nonfinite_grad(adversarial_classifier):
    plan, opt1, opt2, batch = _make_adversarial_plan(
        adversarial_classifier=adversarial_classifier
    )
    device = batch[REGISTRY_KEYS.BATCH_KEY].device
    plan.forward = lambda *args, **kwargs: _finite_forward_return(
        device, plan.module.n_latent
    )

    with patch(
        "scvi.train._trainingplans.torch.nn.utils.clip_grad_norm_",
        side_effect=RuntimeError("nonfinite grad"),
    ):
        result = plan.training_step(batch, 0)

    assert torch.isfinite(result).all()
    opt1.step.assert_not_called()
    if opt2 is not None:
        opt2.step.assert_not_called()


@pytest.mark.parametrize("adversarial_classifier", [False, True])
def test_adversarial_plan_finite_path_steps(adversarial_classifier):
    plan, opt1, opt2, batch = _make_adversarial_plan(
        adversarial_classifier=adversarial_classifier
    )
    device = batch[REGISTRY_KEYS.BATCH_KEY].device
    plan.forward = lambda *args, **kwargs: _finite_forward_return(
        device, plan.module.n_latent
    )

    with patch(
        "scvi.train._trainingplans.torch.nn.utils.clip_grad_norm_",
        return_value=1.0,
    ):
        plan.training_step(batch, 0)

    opt1.step.assert_called_once()
    if opt2 is not None:
        opt2.step.assert_called_once()


@pytest.mark.parametrize("adversarial_classifier", [False, True])
def test_semisupervised_adversarial_plan_skips_nonfinite_loss(adversarial_classifier):
    plan, opt1, opt2, batch = _make_semisupervised_adversarial_plan(
        adversarial_classifier=adversarial_classifier
    )
    device = batch[REGISTRY_KEYS.BATCH_KEY].device

    def forward_with_nan(*args, **kwargs):
        inference_outputs, _, _ = _finite_forward_return(device, plan.module.n_latent)
        scvi_loss = LossOutput(
            loss=torch.tensor(float("nan"), device=device),
            n_obs_minibatch=4,
        )
        return inference_outputs, None, scvi_loss

    plan.forward = forward_with_nan
    result = plan.training_step(batch, 0)

    assert torch.isfinite(result).all()
    opt1.step.assert_not_called()
    if opt2 is not None:
        opt2.step.assert_not_called()


@pytest.mark.parametrize("adversarial_classifier", [False, True])
def test_semisupervised_adversarial_plan_skips_nonfinite_grad(adversarial_classifier):
    plan, opt1, opt2, batch = _make_semisupervised_adversarial_plan(
        adversarial_classifier=adversarial_classifier
    )
    device = batch[REGISTRY_KEYS.BATCH_KEY].device
    plan.forward = lambda *args, **kwargs: _finite_forward_return(
        device, plan.module.n_latent
    )

    with patch(
        "scvi.train._trainingplans.torch.nn.utils.clip_grad_norm_",
        side_effect=RuntimeError("nonfinite grad"),
    ):
        result = plan.training_step(batch, 0)

    assert torch.isfinite(result).all()
    opt1.step.assert_not_called()
    if opt2 is not None:
        opt2.step.assert_not_called()


@pytest.mark.parametrize("adversarial_classifier", [False, True])
def test_semisupervised_adversarial_plan_finite_path_steps(adversarial_classifier):
    plan, opt1, opt2, batch = _make_semisupervised_adversarial_plan(
        adversarial_classifier=adversarial_classifier
    )
    device = batch[REGISTRY_KEYS.BATCH_KEY].device
    plan.forward = lambda *args, **kwargs: _finite_forward_return(
        device, plan.module.n_latent
    )

    with patch(
        "scvi.train._trainingplans.torch.nn.utils.clip_grad_norm_",
        return_value=1.0,
    ):
        plan.training_step(batch, 0)

    opt1.step.assert_called_once()
    if opt2 is not None:
        opt2.step.assert_called_once()


@pytest.mark.parametrize(
    ("current", "n_warm_up", "min_kl_weight", "max_kl_weight", "expected"),
    [
        (0, 400, 0.0, 1.0, 0.0),
        (200, 400, 0.0, 1.0, 0.5),
        (400, 400, 0.0, 1.0, 1.0),
        (0, 400, 0.5, 1.0, 0.5),
        (200, 400, 0.5, 1.0, 0.75),
        (400, 400, 0.0, 2.0, 2.0),
    ],
)
def test_compute_kl_weight_linear_annealing(
    current, n_warm_up, min_kl_weight, max_kl_weight, expected
):
    kl_weight = _compute_kl_weight(current, 1, n_warm_up, None, max_kl_weight, min_kl_weight)
    assert kl_weight == pytest.approx(expected)
    kl_weight = _compute_kl_weight(1, current, None, n_warm_up, max_kl_weight, min_kl_weight)
    assert kl_weight == pytest.approx(expected)


@pytest.mark.parametrize("max_kl_weight", [1.0, 2.0])
def test_compute_kl_weight_no_annealing(max_kl_weight):
    assert _compute_kl_weight(1, 1, None, None, max_kl_weight, 0.0) == max_kl_weight


def test_compute_kl_weight_min_greater_max():
    with pytest.raises(ValueError):
        _compute_kl_weight(1, 1, 400, None, 0.5, 1.0)


@pytest.mark.parametrize(
    ("epoch", "step", "n_epochs_kl_warmup", "n_steps_kl_warmup", "expected"),
    [
        (0, 100, 100, 100, 0.0),
        (50, 200, 100, 1000, 0.5),
        (100, 200, 100, 1000, 1.0),
    ],
)
def test_compute_kl_precedence(epoch, step, n_epochs_kl_warmup, n_steps_kl_warmup, expected):
    kl_weight = _compute_kl_weight(epoch, step, n_epochs_kl_warmup, n_steps_kl_warmup, 1.0, 0.0)
    assert kl_weight == expected


def test_loss_args():
    """Test that self._loss_args is set correctly."""
    adata = synthetic_iid()
    SCVI.setup_anndata(adata)
    vae = SCVI(adata)
    tp = TrainingPlan(vae.module)

    loss_args = [
        "tensors",
        "inference_outputs",
        "generative_outputs",
        "kl_weight",
    ]
    assert len(tp._loss_args) == len(loss_args)
    for arg in loss_args:
        assert arg in tp._loss_args


def test_semisupervisedtrainingplan_metrics():
    adata = scvi.data.synthetic_iid(n_labels=3)
    scvi.model.SCANVI.setup_anndata(
        adata,
        labels_key="labels",
        unlabeled_category="label_0",
        batch_key="batch",
    )
    model = scvi.model.SCANVI(adata)
    model.train(max_epochs=1, check_val_every_n_epoch=1)

    for mode in ["train", "validation"]:
        for metric in [
            METRIC_KEYS.ACCURACY_KEY,
            METRIC_KEYS.F1_SCORE_KEY,
            METRIC_KEYS.CLASSIFICATION_LOSS_KEY,
        ]:
            assert f"{mode}_{metric}" in model.history_
