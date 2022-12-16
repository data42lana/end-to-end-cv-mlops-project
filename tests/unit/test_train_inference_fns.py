import pytest
import mlflow
import torch

from src.train_inference_fns import (precision_recall_fbeta_scores, train_one_epoch, 
                                     eval_one_epoch, predict)
from src.fine_tune_model import run_train

def test_precision_recall_fbeta_scores():
    x1, y1, x2, y2 = 1.0, 1.0, 4.0, 4.0
    gt = {'boxes': torch.tensor([[x1, y1, x2, y2],
                                 [x1+3, y1, x2+3, y2]]), 'labels': torch.tensor([1, 1])}
    pred = {'boxes': torch.tensor([[x1, y1+1, x2, y2+1]]), 'labels': torch.tensor([1]), 'scores': torch.tensor([1])}
    res = precision_recall_fbeta_scores((gt, gt), (pred, pred))
    assert res['precision'] == 1
    assert res['recall'] == 0.5
    assert round(res['f_beta'], 2) == 0.67

@pytest.mark.slow
def test_train_one_epoch(dataloader, frcnn_model):
    frcnn_model.train()
    model_params = [p for p in frcnn_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(model_params, lr=0.0001)
    res = train_one_epoch(dataloader, frcnn_model, optimizer)
    assert len(res['epoch_dict_losses']) == 4
    assert isinstance(res['epoch_loss'], float)

def test_eval_one_epoch(dataloader, frcnn_model):
    frcnn_model.eval()
    res = eval_one_epoch(dataloader, frcnn_model)
    assert len(res['epoch_scores']) == 3
    assert len(res['results']) == len(dataloader.dataset)

def test_predict(img, frcnn_model, tmp_path):
    frcnn_model.eval()
    res = predict(img, frcnn_model, save_predict_path=tmp_path / 'pred.jpg')
    assert isinstance(res, int)
    assert res >= 0

@pytest.mark.slow
class TestRunTrain:

    def test_run_train_results(self, dataloader, frcnn_model): 
        res = run_train(dataloader, dataloader, frcnn_model, 2, 'SGD', {'lr': 0.001})
        assert 'epoch_dict_losses' in res['train_res']
        assert 'epoch_loss' in res['train_res']
        assert 'epoch_scores' in res['eval_res']
        assert 'results' in res['eval_res']

    def test_run_train_save_ckpt_and_random_output(self, dataloader, frcnn_model, tmp_path):
        _ = run_train(dataloader, dataloader, frcnn_model, 2, 'SGD', {'lr': 0.001},
                      save_best_model_path=tmp_path, metric_to_find_best_model='f_beta', 
                      model_name='frcnn', save_best_ckpt=True, 
                      save_random_best_model_output_path=tmp_path)
        assert (tmp_path / 'frcnn_best_f_beta_1_weights_ckpt.pt').exists()
        assert (tmp_path / 'val_outs').exists()
        assert len([p for p in (tmp_path / 'val_outs').iterdir()]) in [1, 2]

    def test_run_train_mlflow_tracking(self, dataloader, frcnn_model, tmp_path): 
        reg_model_name = 'test_run_train_mltracking'
        mlflow.set_tracking_uri(f'sqlite:///{tmp_path}/tmlruns.db')
        exp_id = mlflow.create_experiment('TestRTModel', tmp_path.as_uri())
        with mlflow.start_run(experiment_id=exp_id) as run:
            _ = run_train(dataloader, dataloader, frcnn_model, 2, 'SGD', {'lr': 0.001},
                          metric_to_find_best_model='f_beta', log_metrics=True, 
                          register_best_log_model=True, reg_model_name=reg_model_name)
        client = mlflow.MlflowClient()
        assert client.get_latest_versions(reg_model_name)
        assert client.get_metric_history(run.info.run_id, 'f_beta')
    