import torch
from pet_breed.calibration import ece, fit_temperature, select_threshold


def test_temperature_positive():
    logits = torch.tensor([[2., 0.], [0., 2.], [1., 0.], [0., 1.]])
    labels = torch.tensor([0, 1, 0, 1])
    assert fit_temperature(logits, labels) > 0


def test_temperature_preserves_top_class():
    logits = torch.tensor([[2., 0., 1.], [0., 3., 1.]])
    for t in [0.5, 1., 2.]:
        assert torch.equal(logits.argmax(1), (logits / t).argmax(1))


def test_threshold_selection():
    probs = torch.tensor([[.95, .05], [.9, .1], [.6, .4], [.55, .45]])
    labels = torch.tensor([0, 0, 1, 1])
    policy = select_threshold(probs, labels, target_accuracy=.95, min_coverage=.5)
    assert policy['target_met']
    assert policy['coverage'] >= .5
    assert policy['selective_accuracy'] >= .95


def test_ece_bounds():
    probs = torch.tensor([[.8, .2], [.3, .7]])
    assert 0 <= ece(probs, torch.tensor([0, 1])) <= 1
