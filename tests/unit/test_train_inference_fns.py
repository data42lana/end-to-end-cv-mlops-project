import mlflow
import torch

from src.train_inference_fns import (precision_recall_fbeta_scores, train_one_epoch, 
                                     eval_one_epoch, predict)
from src.fine_tune_model import run_train

def test_precision_recall_fbeta_scores():
    x1, y1, x2, y2 = 1.0, 1.0, 4.0, 4.0
    gts = ({'boxes': torch.tensor([[x1, y1, x2, y2]]), 'labels': torch.tensor([1])}, 
           {'boxes': torch.tensor([[x1, y1, x2, y2],
                                   [x1+2, y1, x2+2, y2]]), 'labels': torch.tensor([1, 1])})
    preds = ({'boxes': torch.tensor([[x1, y1+1, x2, y2+1]]), 'scores': torch.tensor([1])}, 
             {'boxes': torch.tensor([[x1+3, y1, x2+3, y2]]), 'scores': torch.tensor([1])})
    res = precision_recall_fbeta_scores(gts, preds)
    assert res['precision'] == 1
    assert res['recall'] == 0.67
    assert res['f_beta'] == 0.89

def test_train_one_epoch(dataloader, frcnn_model):
    model_params = [p for p in frcnn_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(model_params)
    res = train_one_epoch(dataloader, frcnn_model, optimizer)
    assert len(res['epoch_dict_losses']) == 3
    assert len(res['epoch_loss']) == 1

def test_eval_one_epoch(dataloader, frcnn_model):
    res = eval_one_epoch(dataloader, frcnn_model)
    assert len(res['epoch_scores']) == 3
    assert len(res['results']) == len(dataloader.dataset)

def test_predict(imgbboxdataset, frcnn_model, show_scores=False, device=torch.device('cpu'), 
            save_predict_path=None):
    res = predict(imgbboxdataset[0][0], frcnn_model)
    assert isinstance(res, int)
    assert res >= 0

class TestRunTrain:

    def test_run_train_results(dataloader, frcnn_model, tmp_path): 
        res = run_train(dataloader, dataloader, frcnn_model, 2, 'SGD', {'lr': 0.001},
                        save_best_model_path=tmp_path / 'best_model', 
                        metric_to_find_best_model='f_beta')
        assert 'epoch_dict_losses' in res['train_res']
        assert 'epoch_loss' in res['train_res']
        assert 'epoch_scores' in res['eval_res']
        assert 'results' in res['eval_res']

    def test_run_train_save_ckpt_and_random_output(dataloader, frcnn_model, tmp_path):
        save_path = tmp_path / 'best_model'
        _ = run_train(dataloader, dataloader, frcnn_model, 2, 'SGD', {'lr': 0.001},
                        save_best_model_path=save_path, metric_to_find_best_model='f_beta', 
                        model_name='frcnn', save_best_ckpt=True, 
                        save_random_best_model_output_path=save_path)
        assert (save_path / 'best_frcnn_f_beta_1_weights_ckpt.pt').exists()
        assert (save_path / 'val_outs/epoch_2.jpg')

    def test_run_train_mlflow_tracking(dataloader, frcnn_model, tmp_path, mltracking): 
        reg_model_name = 'best_reg_model'
        # mlflow.set_tracking_uri(f'sqlite:////{tmp_path}/mlruns.db')
        with mlflow.start_run() as run:
            _ = run_train(dataloader, dataloader, frcnn_model, 2, 'SGD', {'lr': 0.001},
                        save_best_model_path=tmp_path / 'best_model', metric_to_find_best_model='f_beta',
                        log_metrics=True, register_best_log_model=True, reg_model_name=reg_model_name)
        client = mlflow.MlflowClient()
        assert client.get_latest_versions(reg_model_name)
        assert client.get_metric_history(run.info.run_id, 'train_epoch_loss')
    