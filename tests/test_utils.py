"""Shared plumbing: LR schedule, resume round-trip, split determinism, subsampling."""

import math
import types

import numpy as np
import pytest
import torch
from train_utils import load_train_state, save_train_state, train_log_path, warmup_cosine_lambda

from tutorial_rs import stratified_split


def test_warmup_cosine_lambda():
    f = warmup_cosine_lambda(total_steps=100, warmup_steps=10)
    assert f(0) == pytest.approx(1 / 11)
    assert f(10) == pytest.approx(1.0)
    assert f(100) == pytest.approx(1 / 50)
    mid = 10 + 45
    assert f(mid) == pytest.approx(1 / 50 + (1 - 1 / 50) * 0.5 * (1 + math.cos(math.pi / 2)))


def test_train_state_round_trip(tmp_path):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_lambda(10, 2))
    optimizer.step(); scheduler.step()
    path = str(tmp_path / "run.train.pt")
    save_train_state(path, step=7, model=model, optimizer=optimizer, scheduler=scheduler,
                     extra={"center": torch.ones(3)})

    fresh = torch.nn.Linear(4, 2)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    fresh_sched = torch.optim.lr_scheduler.LambdaLR(fresh_opt, warmup_cosine_lambda(10, 2))
    step, extra = load_train_state(path, fresh, fresh_opt, fresh_sched, "cpu")
    assert step == 7 and torch.equal(extra["center"], torch.ones(3))
    assert torch.equal(fresh.weight, model.weight)
    assert fresh_sched.last_epoch == scheduler.last_epoch
    assert load_train_state(str(tmp_path / "missing.pt"), fresh, fresh_opt, fresh_sched, "cpu") == (0, {})


def test_train_log_path_is_per_checkpoint():
    assert train_log_path("ckpts/mae_vit_s8.pt") != train_log_path("ckpts/mae_vit_s8_ben.pt")


def test_stratified_split_is_deterministic_and_disjoint():
    dataset = types.SimpleNamespace(targets=np.repeat(np.arange(4), 25).tolist())
    train_a, test_a = stratified_split(dataset)
    train_b, test_b = stratified_split(dataset)
    assert np.array_equal(train_a, train_b) and np.array_equal(test_a, test_b)
    assert not set(train_a) & set(test_a)
    assert len(test_a) == 20
    assert np.bincount(np.asarray(dataset.targets)[test_a]).tolist() == [5, 5, 5, 5]


def test_subsample_per_class():
    from eurosat_probe import subsample_per_class

    labels = np.repeat(np.arange(3), 10)
    picked = subsample_per_class(labels, k=4)
    assert np.bincount(labels[picked]).tolist() == [4, 4, 4]
    assert np.array_equal(picked, subsample_per_class(labels, k=4))
