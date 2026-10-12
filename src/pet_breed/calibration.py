"""Validation-only temperature scaling and selective prediction policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pet_breed.dataset import PetBreedDataset
from pet_breed.inference import PetBreedPredictor


def ece(probs: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    confidence, predicted = probs.max(1)
    correct = predicted.eq(labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    result = 0.0
    for i in range(bins):
        mask = (confidence >= edges[i]) & ((confidence <= edges[i + 1]) if i == bins - 1 else (confidence < edges[i + 1]))
        if mask.any():
            result += float(mask.float().mean() * (correct[mask].mean() - confidence[mask].mean()).abs())
    return result


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit one positive scalar by minimizing validation negative log likelihood."""
    log_t = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    x = logits.double()
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x / log_t.exp(), labels)
        loss.backward()
        return loss
    opt.step(closure)
    temperature = float(log_t.detach().exp())
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature fitting failed')
    return temperature


def select_threshold(probs: torch.Tensor, labels: torch.Tensor, target_accuracy: float = .95, min_coverage: float = .50) -> dict:
    confidence, predictions = probs.max(1)
    correct = predictions.eq(labels)
    candidates = torch.cat((torch.tensor([0.0]), torch.unique(confidence).sort().values))
    for candidate in candidates:
        accepted = confidence >= candidate
        coverage = float(accepted.float().mean())
        if coverage < min_coverage:
            break
        accuracy = float(correct[accepted].float().mean())
        if accuracy >= target_accuracy:
            return {'threshold': float(candidate), 'coverage': coverage, 'selective_accuracy': accuracy, 'target_met': True}
    # Fail closed: do not claim the target was achieved.
    return {'threshold': 1.0, 'coverage': 0.0, 'selective_accuracy': None, 'target_met': False}


def reliability_data(probs: torch.Tensor, labels: torch.Tensor, bins: int = 15):
    conf, pred = probs.max(1)
    correct = pred.eq(labels).float()
    centers, accuracy, mean_confidence = [], [], []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (conf >= lo) & ((conf <= hi) if i == bins - 1 else (conf < hi))
        if mask.any():
            centers.append((lo + hi) / 2)
            accuracy.append(float(correct[mask].mean()))
            mean_confidence.append(float(conf[mask].mean()))
    return centers, accuracy, mean_confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='models/frozen/resnet18_v1.pt')
    parser.add_argument('--classes', default='data/processed/classes.json')
    parser.add_argument('--manifest', default='data/processed/manifest.csv')
    parser.add_argument('--output', default='models/frozen/calibration.json')
    parser.add_argument('--report-dir', default='reports')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--target-accuracy', type=float, default=.95)
    parser.add_argument('--min-coverage', type=float, default=.50)
    args = parser.parse_args()

    predictor = PetBreedPredictor(args.checkpoint, args.classes)
    dataset = PetBreedDataset(args.manifest, split='val', transform=predictor.transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    all_logits, all_labels = [], []
    with torch.inference_mode():
        for images, labels in loader:
            all_logits.append(predictor.model(images).cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).long()
    temperature = fit_temperature(logits, labels)
    before = torch.softmax(logits, dim=1)
    after = torch.softmax(logits / temperature, dim=1)
    policy = select_threshold(after, labels, args.target_accuracy, args.min_coverage)
    metrics = {
        'split': 'val', 'samples': len(dataset), 'temperature': temperature,
        'ece_before': ece(before, labels), 'ece_after': ece(after, labels),
        'nll_before': float(F.cross_entropy(logits, labels)),
        'nll_after': float(F.cross_entropy(logits / temperature, labels)),
        **policy,
    }
    artifact = {'model_version': predictor.model_version, 'temperature': temperature,
                'threshold': policy['threshold'], 'validation': metrics}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + '\n', encoding='utf-8')
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / 'calibration_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], '--', label='Perfect calibration')
    for name, probs in [('Before', before), ('After', after)]:
        _, accuracies, confidences = reliability_data(probs, labels)
        ax.plot(confidences, accuracies, 'o-', label=f'{name} (ECE={ece(probs, labels):.4f})')
    ax.set(xlabel='Mean confidence', ylabel='Observed accuracy', title='Validation reliability diagram', xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / 'calibration.png', dpi=160)
    plt.close(fig)
    print(json.dumps(artifact, indent=2))
    if not policy['target_met']:
        print('WARNING: Target selective accuracy and minimum coverage were NOT jointly achieved on validation.')


if __name__ == '__main__':
    main()
