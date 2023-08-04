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
    Optional, Callable, Union,
    List, Tuple, Dict, Literal
)
from datasets import (
    ModelNet40_train,
    ModelNet40_test, 
    ShapeNetCore_train,
    ShapeNetCore_test
)
from model import AutoEncoder
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
    categories = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc='Getting codewords'):
            point_clouds, lbls, cnames = data
            point_clouds = point_clouds.to(device)
            point_clouds = point_clouds.permute(0, 2, 1)
            bottleneck = autoencoder.encoder(point_clouds)
            codewords.append(bottleneck.cpu().numpy())
            labels.append(lbls)
            categories.append(cnames)

    codewords = np.concatenate(codewords, axis=0)
    labels = np.concatenate(labels)
    categories = np.concatenate(categories).reshape(-1, 1)

    if one_hot:
        one = OneHotEncoder(sparse_output=False)
        categories = one.fit_transform(categories)       

    return codewords, categories
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
def classify(
    classifier: Literal['SVM'],
    x_train: np.ndarray[float],
    y_train: np.ndarray[float],
    x_test: np.ndarray[float],
    y_test: np.ndarray[float]
) -> np.ndarray[float]:
    
    if classifier == 'SVM':
        model = OneVsRestClassifier(svm.LinearSVC(random_state=0, verbose=0, max_iter=100000, dual="auto"))
    else:
        NotImplementedError
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

    # print('Precision:', precision_score(y_test, y_pred, average='micro'))
    # print('Recall:', recall_score(y_test, y_pred, average='micro'))
    # print('F1 Score:', f1_score(y_test, y_pred, average='micro'))

    print(classification_report(y_test, y_pred, digits=4, zero_division=np.nan))

    return model


if __name__ == '__main__':

    train_dataset = ModelNet40_train(
            root='../data/ModelNet40/modelnet40_ply_hdf5_2048/', 
            npoints=2048, 
            normalize=True,
            data_augmentation=False
        )
    test_dataset = ModelNet40_test(
        root='../data/ModelNet40/modelnet40_ply_hdf5_2048/',
        npoints=2048,
        split='test',
        normalize=True
    )
    
    # subset_idxs = np.arange(1024)
    # train_dataset = torch.utils.data.Subset(train_dataset, subset_idxs)
    # test_dataset = torch.utils.data.Subset(test_dataset, subset_idxs)

    train_dataloader = DataLoader(
        train_dataset, batch_size=16, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4
    )

    dev, _ = get_available_devices()

    cdws_train, lbls_train = get_codewords(
        model_ckpt='lightning_logs/version_0/checkpoints/best_checkpoint.ckpt',
        dataloader=train_dataloader,
        dataset_name='ModelNet40',
        device=dev,
        one_hot=True
    )
    cdws_test, lbls_test = get_codewords(
        model_ckpt='lightning_logs/version_0/checkpoints/best_checkpoint.ckpt',
        dataloader=test_dataloader,
        dataset_name='ModelNet40',
        device=dev,
        one_hot=True
    )

    _ = classify(
        classifier="SVM",
        x_train=cdws_train,
        y_train=lbls_train,
        x_test=cdws_test,
        y_test=lbls_test,
    )

