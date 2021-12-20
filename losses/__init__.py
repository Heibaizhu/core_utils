# -*- coding: utf-8 -*-

from .losses import (CharbonnierLoss, GANLoss, L1Loss, MSELoss, SmoothL1Loss,
                     WeightedTVLoss, g_path_regularize, gradient_penalty_loss,
                     r1_penalty, AdaWeightedLoss, CensusLoss, SSIMLoss, EdgeAwareSmoothLoss,
                     MCLoss, PerceptualLoss, GradientPenaltyLoss)

__all__ = [
    'L1Loss', 'MSELoss', 'SmoothL1Loss', 'CharbonnierLoss', 'WeightedTVLoss', 'PerceptualLoss',
    'GANLoss', 'gradient_penalty_loss', 'r1_penalty', 'g_path_regularize',
    'AdaWeightedLoss', 'CensusLoss', 'SSIMLoss', 'EdgeAwareSmoothLoss', 'MCLoss', 'GradientPenaltyLoss'
]
