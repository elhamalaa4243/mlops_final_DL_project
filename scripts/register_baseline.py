"""Register an existing checkpoint as an MLflow experiment run, without training."""
import argparse
import json
from pathlib import Path

import mlflow
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='models/frozen/resnet18_v1.pt')
    parser.add_argument('--calibration', default='models/frozen/calibration.json')
    parser.add_argument('--report-dir', default='reports/resnet18_v1')
    parser.add_argument('--tracking-uri', default='http://127.0.0.1:5000')
    parser.add_argument('--experiment', default='pet-breed-classification')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    calibration_path = Path(args.calibration)
    if not checkpoint_path.is_file() or not calibration_path.is_file():
        raise FileNotFoundError('Checkpoint and calibration.json must both exist')

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
    val = calibration.get('validation', {})
    recorded = ckpt.get('val_metrics', {})

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name='resnet18-existing-baseline') as run:
        mlflow.set_tags({
            'run_type': 'imported_existing_checkpoint',
            'dataset': 'Oxford-IIIT Pet',
            'split_for_calibration': 'val',
            'training_reexecuted': 'false',
        })
        mlflow.log_params({
            'backbone': ckpt.get('model_name', 'resnet18'),
            'num_classes': ckpt.get('num_classes', 37),
            'seed': ckpt.get('seed', 'unknown'),
            'best_epoch': ckpt.get('epoch', 'unknown'),
            'learning_rate': '1e-4 (historical CLI; not stored in checkpoint)',
            'batch_size': '16 (historical CLI; not stored in checkpoint)',
        })
        metrics = {
            'val_accuracy': recorded.get('accuracy'),
            'val_macro_f1': recorded.get('macro_f1'),
            'val_ece_calibrated': val.get('ece_after'),
            'val_ece_uncalibrated': val.get('ece_before'),
            'val_nll_calibrated': val.get('nll_after'),
            'temperature': calibration.get('temperature'),
            'threshold': calibration.get('threshold'),
            'val_coverage': val.get('coverage'),
            'val_selective_accuracy': val.get('selective_accuracy'),
        }
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(name, float(value))
        mlflow.log_artifact(str(checkpoint_path), artifact_path='model')
        mlflow.log_artifact(str(calibration_path), artifact_path='calibration')
        report_dir = Path(args.report_dir)
        if report_dir.is_dir():
            mlflow.log_artifacts(str(report_dir), artifact_path='test_reports')
        for report in ('reports/calibration.png', 'reports/calibration_metrics.json'):
            if Path(report).is_file():
                mlflow.log_artifact(report, artifact_path='calibration')
        print(f'Run ID: {run.info.run_id}')
        print(f'MLflow: {args.tracking_uri}')
        print('Existing checkpoint registered; no training performed.')


if __name__ == '__main__':
    main()
