# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tool pose criteria factories."""

# Third Party
import torch

# CuRobo
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.types.device_cfg import DeviceCfg


def test_linear_motion_uses_requested_device_cfg() -> None:
    """Place factory-created criteria tensors on the requested device."""
    device_cfg = DeviceCfg(device="cpu")

    criteria = ToolPoseCriteria.linear_motion(axis="x", device_cfg=device_cfg)

    ToolPoseCriteria(device_cfg=device_cfg).copy_(criteria)

    assert criteria.device_cfg == device_cfg
    assert criteria.terminal_pose_axes_weight_factor.device == torch.device("cpu")
    assert criteria.non_terminal_pose_axes_weight_factor.device == torch.device("cpu")
    assert criteria.terminal_pose_convergence_tolerance.device == torch.device("cpu")
    assert criteria.non_terminal_pose_convergence_tolerance.device == torch.device("cpu")
    assert criteria.project_distance_to_goal.device == torch.device("cpu")
