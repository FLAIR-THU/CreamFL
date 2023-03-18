import random, torch
import os
import errno


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, torch.autograd.Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    # y_one_hot = y_one_hot.view(y.shape, -1)
    return torch.autograd.Variable(y_one_hot) if isinstance(y, torch.autograd.Variable) else y_one_hot


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_txt_dict():
    TXT_dict = {}
    return TXT_dict


def data_dict_reader(data, phase):
    data_dict = get_txt_dict()
    if phase == 'train':
        txt_path = data_dict[data][0]
    else:
        txt_path = data_dict[data][1]
    data_dict = {}
    with open(txt_path) as f:
        txt_lines = f.readlines()
    label_id = -1
    img_list = []
    for _, value in enumerate(txt_lines):
        info_list = value.split(' ')
        this_label_id = int(info_list[1])
        if label_id != int(this_label_id):
            if label_id != -1:
                data_dict[this_dict_name] = img_list
            label_id = this_label_id
            this_dict_name = 'class_tra_' + str(label_id).zfill(2)
            img_list = []
            img_list.append(info_list[0])
        else:
            img_list.append(info_list[0])
    data_dict[this_dict_name] = img_list
    return data_dict


def norml2(vec):  # input N by F
    F = vec.size(1)
    w = torch.sqrt((torch.t(vec.pow(2).sum(1).repeat(F, 1))))
    return vec.div(w)


def createID(num_int, Len, N):
    """uniformly distributed"""
    multiple = N // num_int
    remain = N % num_int
    if remain != 0: multiple += 1

    ID = torch.zeros(N, Len)
    for i in range(Len):
        idx_all = []
        for _ in range(multiple):
            idx_base = [j for j in range(num_int)]
            random.shuffle(idx_base)
            idx_all += idx_base

        idx_all = idx_all[:N]
        random.shuffle(idx_all)
        ID[:, i] = torch.Tensor(idx_all)

    return ID


def get_data_by_txt(txt_path):
    data_gen = []
    label = []
    with open(txt_path) as f:
        txt_lines = f.readlines()
    for key, value in enumerate(txt_lines):
        info_list = value.split(' ')
        data_gen.append(info_list[0])
        label.append(int(info_list[1]))
    return data_gen, label


def get_query_info_by_txt(txt_path):
    data, label = get_data_by_txt(txt_path)
    query_id, retrieval_list = get_info_by_label(label)
    return data, query_id, retrieval_list


def get_info_by_label(label):
    query_id = []
    unique_list = list(set(label))
    retrieval_list = []
    label_dict = {}
    for i in unique_list:
        if i in label_dict:
            continue
        else:
            label_dict.update({i: [idx for idx, x in enumerate(label) if x == i]})
    for key, value in enumerate(label):
        gt = label_dict[value]
        query_id.append(key)
        retrieval_list.append([gt, [key]])
    return query_id, retrieval_list


def data_info(data, phase):
    data_dict = get_txt_dict()
    if phase == 'train':
        txt_path = data_dict[data][0]
    else:
        txt_path = data_dict[data][1]
    return get_query_info_by_txt(txt_path)
