#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utilities for resuming training from checkpoints."""


def resolve_resume_epoch(saved_epoch: int) -> int:
    """
    Convert the epoch stored in a checkpoint to the epoch to start from when resuming.

    The training script saves a checkpoint at the end of each epoch, where the stored
    epoch number corresponds to the completed epoch. When resuming, training should
    continue from that same epoch (not epoch + 1).
    """
    if saved_epoch <= 0:
        return 1
    return saved_epoch
