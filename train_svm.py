from datetime import datetime
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize, OneHotEncoder
from sklearn.metrics import *
import pytorch_lightning as pl
from tqdm import tqdm
from typing import (
    Iterable, Optional, Callable, 
    Union, List, Tuple, 
    Dict, Literal, Callable
)
from datasets import (
    ModelNet40_train,
    ModelNet40_test, 
    ShapeNetCore_train,
    ShapeNetCore_test
)
from model import AutoEncoder
from utils import to_one_hots
from train_utils import get_available_devices


#-------------------------------------------------------------------------------
def get_codewords(
        model_ckpt: str,
        dataloader: torch.utils.data.DataLoader,
        dataset_name: Literal['ModelNet40', 'ShapeNetCore'],
        device: torch.device,
        one_hot: Optional[bool] = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Feed data into the Encoder network to get codewords associated
    to point clouds.
    
    Parameters:
    ----------- 
        model_ckpt: (str)
            Path to the checkpoint of the model we want to use.

        dataloader: (torch.utils.data.DataLoader)
            The dataloader containing the dataset used to produce the codewords.
        
        device: (torch.device)
            The device on which data are processed.

        one_hot: (Optional[bool], default = True)
            If `True`, labels are extracted from the dataloader and converted to 
            one-hot encoding.
    
    Returns:
    --------
        codewords: (torch.Tensor) 
            A tensor storing the computed codewords.

        labels: (torch.Tensor)
            A tensor storing the label extracted from the dataloader.
    """

    assert dataset_name in ['ModelNet40', 'ShapeNetCore'], f"Dataset name {dataset_name} not present among ['ModelNet40', 'ShapeNetCore']"

    # Create model and load checkpoints
    autoencoder = AutoEncoder(dataset_name).load_from_checkpoint(model_ckpt).to(device)
    autoencoder.eval()
    
    # Get data and let them through the Encoder
    codewords = []
    labels = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc='Getting codewords'):
            point_clouds, lbls = data
            point_clouds = point_clouds.to(device)
            point_clouds = point_clouds.permute(0, 2, 1)
            bottleneck = autoencoder.encoder(point_clouds)
            codewords.append(bottleneck.cpu().numpy())
            labels.append(lbls)

    codewords = np.concatenate(codewords, axis=0)
    labels = np.concatenate(labels)

    if one_hot:
        one = OneHotEncoder()
        labels = one.fit(labels)       

    return codewords, labels


def load_data(split='train', one_hot=False):
    if one_hot:
        x, y = np.load('data/{}_x.npy'.format(split)), np.load('data/{}_y_onehot.npy'.format(split))
    else:
        x, y = np.load('data/{}_x.npy'.format(split)), np.load('data/{}_y.npy'.format(split))
    return x, y


if __name__ == '__main__':

    # # data preprocessing
    # train_x, train_y = load_data(split='train')
    # test_x, test_y = load_data(split='test')
    # train_y = label_binarize(train_y, classes=range(40))
    # test_y = label_binarize(test_y, classes=range(40))
    # print(train_x.shape, train_y.shape)
    # print(test_x.shape, test_y.shape)

    # model = OneVsRestClassifier(svm.LinearSVC(random_state=0, verbose=1, max_iter=10000))
    # start_time = datetime.now()
    # model.fit(train_x, train_y)
    # print('\033[32mFinish training SVM. It cost totally {} s.\033[0m'.format((datetime.now() - start_time).total_seconds()))

    # y_pred = model.predict(test_x)
    # print('\033[32mAccuracy Overall: {}\033[0m'.format(np.sum(test_y.argmax(axis=1) == y_pred.argmax(axis=1)) / test_x.shape[0]))
    # confusion_matrix(test_y.argmax(axis=1), y_pred.argmax(axis=1))

    # print('Precision:', precision_score(test_y, y_pred, average='micro'))
    # print('Recall:', recall_score(test_y, y_pred, average='micro'))
    # print('F1 Score:', f1_score(test_y, y_pred, average='micro'))

    # print(classification_report(test_y, y_pred, digits=4))

    # joblib.dump(model, 'log/LinearSVC.pkl')

    train_dataset = ModelNet40_train(
            root='../data/ModelNet40/modelnet40_ply_hdf5_2048/', 
            npoints=2048, 
            normalize=True,
            data_augmentation=False
        )

    train_dataloader = DataLoader(
        train_dataset[:128], batch_size=16, shuffle=False, num_workers=4
    )

    dev, _ = get_available_devices()

    cdws, lbls = get_codewords(
        model_ckpt='lightning_logs/version_0/checkpoints/best_checkpoint.ckpt',
        dataloader=train_dataloader,
        dataset_name='ModelNet40',
        device=dev,
        one_hot=True
    )

    print(cdws)
    print(lbls)
