import torch


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, path='best_model.pth'):
        """
        Args:
            patience (int): Validation Loss가 개선되지 않아도 기다리는 최대 Epoch 수
            min_delta (float): 개선으로 간주되는 최소 변화량
            path (str): 최적의 모델 가중치를 저장할 파일 경로
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        """현재 모델을 지정한 경로에 저장"""
        torch.save(model.state_dict(), self.path)

    def load_best_model(self, model):
        """최적의 모델 가중치를 불러옴"""
        model.load_state_dict(torch.load(self.path))