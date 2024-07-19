# preprocess, load and change the distribution of the data
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data import ConcatDataset, Dataset, TensorDataset
import copy
import random
from typing import List
from collections import Counter

streaming_history_data = []
dataset = None
dataset_index = []

def Target_streaming_data_get(
    dataset_index,
    dataset,
    his_data_index,
    streaming_type=0,
    select_num=100,
    client_id=-1,
):
    data_index = []
    targets = [dataset[i][1] for i in dataset_index]
    if len(his_data_index) == 0:
        data_index = np.random.choice(dataset_index, select_num, replace=False)
    if streaming_type == 0:  # huge change in data distribution
        if len(his_data_index) == 0:
            labels = list(set(targets))
            labels_chosen = np.random.choice(
                labels, int(len(labels) / 2), replace=False
            )
            labels = targets
            idxx = []
            idxs_labels = np.argsort(labels)
            for iddx in idxs_labels:
                if labels[iddx] in labels_chosen:
                    idxx.append(iddx)
            data_index = np.array(np.random.choice(idxx, select_num, replace=False))
        else:
            types = []
            labels = targets
            for index in his_data_index:
                types.append(labels[index])

            counter1 = Counter(labels)
            counter2 = Counter(types)
            # 执行减法运算
            result_counter = counter1 - counter2
            # 将结果转换回列表
            index_selected = list(result_counter.elements())
            # index_selected = list((labels) - (types))
            if len(index_selected) < select_num:
                data_index = np.random.choice(dataset_index, select_num, replace=False)
            else:
                index_selected = set(index_selected)
                idxs = []
                idxs_labels = np.argsort(labels)
                for iddx in idxs_labels:
                    if labels[iddx] in index_selected:
                        idxs.append(iddx)
                rand_set = np.random.choice(idxs, select_num, replace=False)
                data_index = np.array(rand_set)
    elif streaming_type == 1:  # past coming again
        if len(his_data_index) == 0 or client_id == -1:
            data_index = np.random.choice(dataset_index, select_num, replace=False)
        his_data = streaming_history_data[client_id]
        if len(his_data) > 3 and random.random() > 0.4:
            data_index = his_data[np.random.randint(0, len(his_data) - 1)]
        else:
            data_index = np.random.choice(dataset_index, select_num, replace=False)
    else:
        print("No such streaming type, return random choice:")
        data_index = np.random.choice(dataset_index, select_num, replace=False)
    return data_index  # return the index of dataset


def Next_stream_data(
    train_dataset, config, client_id=-1
):  # return Train_Dataloader, Test_Dataloader
    global dataset, dataset_index, streaming_history_data
    if dataset is None:  # Initialize
        dataset = train_dataset
        dataset_index = [i for i in range(len(dataset))]
        streaming_history_data = []
    his_index_list = (
        streaming_history_data[-1] if len(streaming_history_data) > 0 else []
    )
    data_now = Target_streaming_data_get(
        dataset_index,
        dataset,
        his_index_list,
        config.streaming_way,
        config.select_num,
        client_id,
    )
    streaming_history_data.append(data_now)
    data_now_local_test = np.random.choice(
        copy.deepcopy(data_now), int(len(data_now) / 5), replace=False
    )
    data_now_local_train = np.array(list(set(data_now) - set(data_now_local_test)))
    return DataLoader(
        Subset(train_dataset, data_now_local_train),
        batch_size=config.batch_size,
        shuffle=True,
    ), DataLoader(
        Subset(train_dataset, data_now_local_test),
        batch_size=config.batch_size,
        shuffle=False,
    )


def dirichlet_data_partition(
    dataset_ini: Dataset, num_clients=10, alpha=0.5
) -> List[Dataset]:
    """split dataset into num_clients clusters using dirichle distribution

    Args:
        dataset_ini (Dataset): the initial dataset
        num_clients (int, optional): Defaults to 10.
        alpha (float, optional): dirichlet distribution alpha. Defaults to 0.5.

    Returns:
        list[Dataset]: the dataset of every client
    """

    # 1. extract labels of the dataset
    if hasattr(dataset_ini, "targets"):
        labels = dataset_ini.targets
    elif hasattr(dataset_ini, "labels"):
        labels = dataset_ini.labels
    else:
        print("No targets attribute, try tensors[1]")
        labels = dataset_ini.tensors[1]
        # raise ValueError("Dataset must have 'targets' or 'labels' attribute")

    labels = np.array(labels)
    num_classes = len(np.unique(labels))

    # 使用Dirichlet分布生成每个类在每个客户端的比例
    distribution = np.random.dirichlet([alpha] * num_clients, num_classes)

    # 确保每个客户端至少有一个样本
    while (distribution.min(axis=1) == 0).any():
        distribution = np.random.dirichlet([alpha] * num_clients, num_classes)

    # 2. assign the data samples to different client
    client_indices = [[] for _ in range(num_clients)]
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        proportions = distribution[k]
        # 计算每个类别按比例分配给每个客户端的索引
        idx_k_split = np.split(
            idx_k, np.cumsum([int(p * len(idx_k)) for p in proportions[:-1]])
        )
        for client in range(num_clients):
            client_indices[client].extend(idx_k_split[client].tolist())
    # 3. return subset of each client
    return [Subset(dataset_ini, indices=idx) for idx in client_indices]
