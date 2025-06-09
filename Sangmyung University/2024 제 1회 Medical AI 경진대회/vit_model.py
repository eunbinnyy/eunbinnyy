import random
import pandas as pd
import numpy as np
import os
import re
import glob
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#!pip install timm

# Configuration 파라미터 정의 이미지 , 에폭 , 학습률 , 배치 , 시드
CFG = {
    'IMG_SIZE':224,
    'EPOCHS':50,
    'LEARNING_RATE':1e-4,
    'BATCH_SIZE':32,
    'SEED':41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

os.chdir('/content/drive/MyDrive/MedicalAI')

# 데이터 전처리 , train , val
# train_label_vec : 이미지 벡터값
df = pd.read_csv('./train.csv')

train_len = int(len(df) * 0.8)
train_df = df.iloc[:train_len]
val_df = df.iloc[train_len:]

train_label_vec = train_df.iloc[:,2:].values.astype(np.float32)
val_label_vec = val_df.iloc[:,2:].values.astype(np.float32)

CFG['label_size'] = train_label_vec.shape[1]

# 커스텀 데이터 셋
# 이미지 , 라벨 , 변환 함수
class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_path_list[index]

        image = cv2.imread('./'+img_path)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_path_list)

# Compose : 이미지 변환 파이프라인 함수 , Normalize : ImageNet 데이터셋에서 일반적으로 쓰는 평균과 표준편차로 RGB 정규화 , ToTensorV2() : 파이토치 텐서로 변환
# 크롭은 ... 안될듯
train_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.HorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평으로 뒤집음
                            A.VerticalFlip(p=0.5),  # 50% 확률로 이미지를 수직으로 뒤집음
                            A.Rotate(limit=(-90, 90), p=0.5),  # -90도에서 90도 사이로 무작위 회전

                            #A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),  # 50% 확률로 색상, 채도, 밝기 조정

                            # 정규화
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])
test_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

train_dataset = CustomDataset(train_df['path'].values, train_label_vec, train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

val_dataset = CustomDataset(val_df['path'].values, val_label_vec, test_transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class BaseModel(nn.Module):
    def __init__(self, gene_size=CFG['label_size']):
        super(BaseModel, self).__init__()

        # ViT 백본 설정 (pretrained 모델 불러오기)
        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

        # ViT의 출력 크기는 hidden_size로 지정됨 (Base model의 hidden_size는 768)
        self.regressor = nn.Linear(self.backbone.config.hidden_size, gene_size)

    def forward(self, x):
        # ViT 모델을 통해 이미지 피처 추출
        outputs = self.backbone(pixel_values=x)  # ViT는 'pixel_values'를 입력으로 받음
        x = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰의 출력 (첫 번째 위치)

        # 회귀 계층을 통과해 최종 출력
        x = self.regressor(x)
        return x

from os.path import exists
if not exists('fmix.zip'):
    !wget -O fmix.zip https://github.com/ecs-vlc/fmix/archive/master.zip
    !unzip -qq fmix.zip
    !mv FMix-master/* ./
    !rm -r FMix-master
from fmix import  sample_mask

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data, targets

def fmix(data, targets, alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)
    #mask =torch.tensor(mask, device=device).float()
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    x1 = torch.from_numpy(mask).to(device)*data
    x2 = torch.from_numpy(1-mask).to(device)*shuffled_data
    targets=(targets, shuffled_targets, lam)

    return (x1+x2), targets

def mixup(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.7)
    data = lam*data + (1-lam)*shuffled_data
    targets = (target, shuffled_target, lam)

    return data, targets

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): 개선이 없을 때 몇 epoch을 기다릴지 (기본값: 7)
            verbose (bool): True일 경우, 개선될 때마다 메시지 출력
            delta (float): 개선으로 간주될 최소 변화 값 (기본값: 0)
            path (str): 모델을 저장할 경로 (기본값: 'checkpoint.pt')
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Validation loss가 감소하면 모델 저장'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.MSELoss().to(device)



    best_loss = 99999999
    best_model = None

    # EarlyStopping 객체 생성
    early_stopping = EarlyStopping(patience=7, verbose=True, path='checkpoint.pt')

    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            #Cutmix , fmix적용
            mix_decision = np.random.rand()
            if mix_decision < 0.25:
                imgs, labels = cutmix(imgs, labels, 1.)
            elif mix_decision >=0.25 and mix_decision < 0.5:
                imgs, labels = fmix(imgs, labels, alpha=1., decay_power=5., shape=(CFG['IMG_SIZE'],CFG['IMG_SIZE']))


            optimizer.zero_grad()
            output = model(imgs.float())

            #Cutmix , fmix loss
            if mix_decision < 0.5:
                loss = criterion(output, labels[0]) * labels[2] + criterion(output, labels[1]) * (1. - labels[2])
            else:
                loss = criterion(output, labels)

            # MSE loss 계산
            #loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        _val_loss = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}]')

        if scheduler is not None:
            scheduler.step(_val_loss)

        if best_loss > _val_loss:
            best_loss = _val_loss
            best_model = model

        # Early Stopping 적용
        early_stopping(_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 조기 종료로 학습이 끝난 후 마지막 체크포인트 모델 로드
    model.load_state_dict(torch.load(early_stopping.path))

    return best_model

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            pred = model(imgs)

            loss = criterion(pred, labels)

            val_loss.append(loss.item())

        _val_loss = np.mean(val_loss)

    return _val_loss

model = BaseModel()
model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)

infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

import torch
from tqdm import tqdm

test_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),

                            A.HorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평으로 뒤집음

                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])


test = pd.read_csv('./test.csv')

test_dataset = CustomDataset(test['path'].values, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4)


def inference_tta(model, test_loader, device, num_tta=5):
    model.eval()
    preds = []

    with torch.no_grad():
        for imgs in tqdm(test_loader):
            imgs = imgs.to(device).float()
            tta_preds = []

            # TTA 적용
            for _ in range(num_tta):
                # 이미지 증강 후 예측 수행
                pred = model(imgs)
                tta_preds.append(pred.detach().cpu())

            # TTA 결과 평균
            tta_preds = torch.mean(torch.stack(tta_preds), dim=0)
            preds.append(tta_preds)

    # 결과를 하나의 배열로 결합
    preds = torch.cat(preds).numpy()

    return preds

# TTA 적용한 예측
preds = inference_tta(infer_model, test_loader, device, num_tta=5)

submit = pd.read_csv('./sample_submission.csv')
submit.iloc[:, 1:] = np.array(preds).astype(np.float32)
submit.to_csv('./sample_vit.csv', index=False)
