# coding:utf-8

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from enum import Enum
from NetworkFunctions import ReLU, ActivateFunction, softmax
import cv2
import os, random, math
from typing import Callable
from tqdm import tqdm
import time
import configparser


class Tag(Enum):
    '''
    描述一个样本的类别，包括类别名称和onehot编码
    '''
    @staticmethod
    def _get_one_hot(number) -> np.ndarray:
        if number not in list(range(10)):
            return np.zeros(10, dtype=np.uint8)
        res = np.zeros(10, dtype=np.uint8)
        res[number] = 1
        return res
    
    def __init__(self, num_value:int, onehot:np.ndarray):
        self.num_value = num_value
        self.onehot = onehot

    ZERO = (0, _get_one_hot(0))
    ONE = (1, _get_one_hot(1))
    TWO = (2, _get_one_hot(2))
    THREE = (3, _get_one_hot(3))
    FOUR = (4, _get_one_hot(4))
    FIVE = (5, _get_one_hot(5))
    SIX = (6, _get_one_hot(6))
    SEVEN = (7, _get_one_hot(7))
    EIGHT = (8, _get_one_hot(8))
    NINE = (9, _get_one_hot(9))

    @classmethod
    def get_by_value(cls, num_value) -> Tag:
        for member in cls:
            if member.num_value == num_value:
                return member
        else:
            raise ValueError(f"Value {num_value} not in Tag!")


class SourceSample:
    '''
    一个处理好的样本数据
    '''
    def __init__(self, length:int, data:np.ndarray, tag:Tag):
        if len(data) != length:
            raise ValueError(f"data size not match! required: {length}, get{len(data)}")
        self.data = data
        self.tag = tag



class Layer:
    METHOD_INIT_RANDOM = 0
    METHOD_INIT_FIX = 1

    def __init__(self, length:int, prev_length:int, matrix_init_method:int, bias_init_method:int, matrix_init_seed = 1, bias_init_seed = 1, is_input:bool = False, layer_value:np.ndarray = None):
        if is_input == True:
            self.layer_value = layer_value
            
        else:
            # 初始化权重矩阵,默认随机
            if matrix_init_method == self.METHOD_INIT_FIX:
                self.weight_matrix = np.full((length, prev_length), matrix_init_seed, dtype=np.float32)
            else:
                self.weight_matrix = (matrix_init_seed * 2 * np.random.ranf((length, prev_length)) - matrix_init_seed).astype(np.float32)

            # 初始化偏置向量，默认随机
            if bias_init_method == self.METHOD_INIT_FIX:
                self.bias_vector = np.full((length, 1), bias_init_seed, dtype=np.float32)
            else:
                self.bias_vector = (bias_init_seed * 2 * np.random.ranf((length, 1)) - bias_init_seed).astype(np.float32)
            
            self.layer_pure_value = np.ndarray = None
            self.layer_value:np.ndarray = None
        
        self.prev_layer:Layer = None
        self.next_layer:Layer = None
        self.layer_gradient:np.ndarray = None
        self.bias_gradient:np.ndarray = None
        self.weight_gradient:np.ndarray = None
        self.activation_gradient = None
        # 分批梯度下降时，存储真正的全量数据，每次slice取出一部分放入layer_value
        self.full_layer_value = None
        self.slice_st = 0
        self.slice_step = 0

def load_samples_from_dir(path:str, target_shape = None, file_filter:str = ".jpg", color_reverse:bool = False) -> tuple:
    '''
    从外部文件加载样本,并返回初始层,初始层应包含一个像素行，样本数列的输入矩阵，和一个分类行，样本数列的
    '''
    file_list = tuple(filter(lambda x : x.lower().endswith(file_filter), os.listdir(path)))

    file_list = [(x, path + os.path.sep + x) for x in file_list]

    datas = []
    trues = []
    

    for file_name in tqdm(file_list):
        img:np.ndarray = cv2.imread(file_name[1], cv2.IMREAD_GRAYSCALE)
        if target_shape != None:
            img = cv2.resize(img, target_shape)
        data_arr = img.flatten().astype(np.float32).reshape(-1,1)
        if color_reverse == True:
            data_arr = 255 - data_arr
        real_num = int(file_name[0].split(".")[0])
        datas.append(data_arr)
        trues.append(Tag._get_one_hot(real_num).reshape(-1,1))

    datas_np = np.concatenate(datas, axis=1) / 255
    trues_np = np.concatenate(trues, axis=1)
    return (datas_np, trues_np)

def load_sample_from_file(file_path:str, color_reverse:bool = False) -> np.ndarray:
    '''
    从外部文件加载样本,并返回标准化的列向量
    '''
    if os.path.isfile(file_path) == False:
        raise RuntimeError("Input not a file")
    
    img:np.ndarray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28,28))
    if color_reverse == True:
        img = 255 - img
    data_arr = img.flatten().astype(np.float32).reshape(-1,1) / 255
    
    return data_arr

class OneHotFullLinkNetwork:
    '''
    网络类，包含了网络所需的各种参数和操作
    '''

    def __init__(self, mid_layer_sizes:list, activation:str="ReLU", learn_rate = 0.002, mid_normal_max = 5, accept_confidence = 0.99, sample_shape=(28,28), train_color_reverse:bool = False):
        self.mid_layer_sizes = mid_layer_sizes
        self.activation = activation
        self.learn_rate = learn_rate
        self.is_inited = False
        self.forward_finish = False
        self.trues_matrix:np.ndarray = None
        self.output_len:int = None
        # layers包含输入层和输出层
        self.layers:List[Layer] = []
        self.accuracy = 0
        self.ensure_99_precent = 0
        self.mid_normal_max = mid_normal_max
        self.accept_confidence = accept_confidence
        self.sample_shape = sample_shape
        self.loss = 0
        self.train_color_reverse = train_color_reverse

    @staticmethod
    def get_n_weight_name(x):
        return f"weight_{x}"
    @staticmethod
    def get_n_bias_name(x):
        return f"bias_{x}"

    def load_samples_and_init_network(self, path:str):
        '''
        装载输入参数，准备各层的参数
        '''
        # 获取样本矩阵、真值矩阵，存入成员变量
        datas_np, trues_np = load_samples_from_dir(path, target_shape = self.sample_shape, color_reverse=self.train_color_reverse)
        
        self.trues_bak:np.ndarray = trues_np
        self.output_len:int = self.trues_bak.shape[0]

        # 样本数据的形状
        input_len:int = datas_np.shape[0]

        # 包含输入和输出层的层神经元数量列表
        layer_sizes = [input_len] + self.mid_layer_sizes + [self.output_len]

        # 创建输入层
        self.layers.append(Layer(input_len, None, None, None, None, None, is_input = True, layer_value=datas_np))
        self.layers[0].full_layer_value = self.layers[0].layer_value.copy()

        # 创建隐藏层和输出层
        for i in range(1, len(layer_sizes)):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i - 1], Layer.METHOD_INIT_RANDOM, bias_init_method=Layer.METHOD_INIT_RANDOM, matrix_init_seed=1, bias_init_seed=1, is_input=False))
        
        # 创建每一层的级联关系
        for i in range(len(self.layers)):
            if i < len(self.layers) - 1:
                self.layers[i].next_layer = self.layers[i + 1]
            if i > 0:
                self.layers[i].prev_layer = self.layers[i - 1]

        # 最后一层的下一层为None
        self.layers[-1].next_layer = None

        # 表示初始化完成
        self.is_inited = True
        self.backward_finish = True

    def init_network_for_predict(self, layers_weight_arrays:list, layers_bias_arrays:list, output_len:int = 10):
        self.output_len = output_len
        input_len = self.sample_shape[0] * self.sample_shape[1]
        layer_sizes = [input_len] + self.mid_layer_sizes + [self.output_len]

        self.layers.append(Layer(input_len, None, None, None, None, None, True, None))

        for i in range(1, len(layer_sizes)):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i - 1], Layer.METHOD_INIT_FIX, bias_init_method=Layer.METHOD_INIT_FIX, matrix_init_seed=1, bias_init_seed=1, is_input=False))

        # 创建每一层的级联关系,填充矩阵
        for i in range(len(self.layers)):
            if i < len(self.layers) - 1:
                self.layers[i].next_layer = self.layers[i + 1]
            if i > 0:
                self.layers[i].prev_layer = self.layers[i - 1]
                self.layers[i].weight_matrix = layers_weight_arrays[i - 1]
                self.layers[i].bias_vector = layers_bias_arrays[i - 1]


        self.layers[-1].next_layer = None
        self.is_inited = True
        self.backward_finish = True
    
        pass

        

    def forward_broadcasting(self, slice_st:int = 0, slice_step:int = -1):
        '''
        执行一次正向传播
        正向传播过程：
        输入层乘以下一层的权重矩阵，得到的值加上下一层的偏置向量，得到的值再执行ReLU，作为下一层的神经元值
        如果下一层是输出层，则ReLU替换为softmax
        '''
        if self.is_inited == False or self.backward_finish == False: 
            raise RuntimeError("Network not inited or backward not finished, cannot do forward broadcasting")
        
        layer = self.layers[0]

        # 每个ephoc之前打乱总样本
        if slice_st == 0:
            total_shf_permu = np.random.permutation(layer.full_layer_value.shape[1])
            layer.full_layer_value = layer.full_layer_value[:,total_shf_permu]
            self.trues_bak = self.trues_bak[:, total_shf_permu]
        
        
        if slice_step == -1:
            slice_step = layer.full_layer_value.shape[1] - slice_st
        slice_step = min(slice_step, self.layers[0].full_layer_value.shape[1] - slice_st)

        layer.layer_value = layer.full_layer_value[:,slice_st:slice_st + slice_step]
        self.trues_matrix = self.trues_bak[:, slice_st:slice_st + slice_step]

        layer.slice_st = slice_st
        layer.slice_step = slice_step
        while layer.next_layer != None:
            next_layer:Layer = layer.next_layer
            next_layer.layer_pure_value = (next_layer.weight_matrix @ layer.layer_value) + next_layer.bias_vector

            if next_layer.next_layer is None:
                next_layer.layer_value = softmax(next_layer.layer_pure_value)
            else:
                if self.activation == "ReLU":
                    next_layer.layer_value = np.copy(next_layer.layer_pure_value)
                    next_layer.layer_value[next_layer.layer_value < 0] = 0
                    #层输出标准化,因为已经经过ReLU，所以直接除以最大值即可，注意将最大值是0的列，最大值改成1避免除以0
                    if self.mid_normal_max != -1:
                        col_max = next_layer.layer_value.max(axis=0)
                        col_max[col_max == 0] = 1
                        next_layer.layer_value  = next_layer.layer_value / col_max * self.mid_normal_max
                    next_layer.activation_gradient = np.copy(next_layer.layer_pure_value)
                    next_layer.activation_gradient[next_layer.activation_gradient <= 0] = 0
                    next_layer.activation_gradient[next_layer.activation_gradient > 0] = 1
            layer = next_layer

        predict_pos = self.layers[-1].layer_value.argmax(axis = 0)
        true_pos = self.trues_matrix.argmax(axis=0)

        self.accuracy = (predict_pos == true_pos).sum() / slice_step
        output_maxes = np.max(self.layers[-1].layer_value, axis=0)
        self.ensure_99_precent = sum(output_maxes > self.accept_confidence) / len(output_maxes)
        self.loss = np.average(-np.log(self.layers[-1].layer_value[true_pos, np.arange(self.layers[-1].layer_value.shape[1])]))
        self.forward_finish = True
        self.backward_finish = False


    def backward_broadcasting(self):
        '''
        执行一次反向传播
        计算最后一层的梯度向量
        计算最后一层权重矩阵的梯度向量
        计算最后一层偏置矩阵的梯度向量
        计算上一层的梯度向量
        同上
        '''
        if self.forward_finish == False:
            raise RuntimeError("Backward broadcasting must after forward broadcasting!")
        
        # 计算最后一层的梯度向量
        cur_layer = self.layers[-1]
        cur_layer.layer_gradient = cur_layer.layer_value - self.trues_matrix

        # 从输出层向上遍历到第一个隐藏层，计算梯度矩阵和偏置梯度矩阵，和各层的梯度
        while cur_layer.prev_layer != None:
            prev_layer = cur_layer.prev_layer
            cur_layer.bias_gradient = np.average(cur_layer.layer_gradient, axis=1).reshape(-1,1).astype(np.float32)
            cur_layer.weight_gradient = ((cur_layer.layer_gradient @ prev_layer.layer_value.T) / cur_layer.layer_gradient.shape[1]).astype(np.float32)
            # 第一层不用算层梯度
            if prev_layer.prev_layer != None:
                prev_layer.layer_gradient = (cur_layer.weight_matrix.T @ cur_layer.layer_gradient) * prev_layer.activation_gradient
            cur_layer = prev_layer

        # 根据梯度矩阵更新权重矩阵和偏置矩阵
        for layer in self.layers[1:]:
            layer.weight_matrix = layer.weight_matrix - self.learn_rate * (layer.weight_gradient)
            layer.bias_vector = layer.bias_vector - self.learn_rate * (layer.bias_gradient)
            # layer.weight_gradient = None
            # layer.bias_gradient = None
            # layer.activation_gradient = None
            # layer.layer_gradient = None

        self.forward_finish = False
        self.backward_finish = True

    def _predict_shaped_array(self, array:np.ndarray):
        unit = array.copy()

        for layer in self.layers[1:]:
            unit = layer.weight_matrix @ unit + layer.bias_vector
            if layer.next_layer != None:
                unit[unit < 0] = 0
            else:
                unit = softmax(unit)

        return unit
    
    def predict_test(self, file_path, train_color_reverse:bool = False):
        data_arr = load_sample_from_file(file_path, train_color_reverse)

        return self._predict_shaped_array(data_arr)
    
    def predict_single_img(self, img:np.ndarray, color_reverse:bool = False):
        # 二值化处理
        _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)

        binary_img = cv2.resize(binary_img, (self.sample_shape))
        if color_reverse == True:
            binary_img = 255 - binary_img
        norm_bin_img = binary_img.flatten().astype(np.float32).reshape(-1,1) / 255

        softmax_predict = self._predict_shaped_array(norm_bin_img)

        return np.argmax(softmax_predict)
    
    def save_config(self, config_dir:str = "nn_config"):
        if os.path.exists(config_dir) == False:
            os.makedirs(config_dir)
        config_file_name = config_dir + os.path.sep + "network_args.ini"

        # 超参数
        config = configparser.ConfigParser()
        config["main"] = {
             "learn_rate":"0.05",
             "mid_normal_max":"5",
             "sample_width":"28",
             "sample_height":"28"
        }
    
        with open(config_file_name, "w") as config_file:
            config.write(config_file)

        # 存储网络参数：每层的权重矩阵、偏置矩阵
        weights_file_name = config_dir + os.path.sep + "weights.npz"
        bias_file_name = config_dir + os.path.sep + "biases.npz"

        weights = {}
        biases = {}

        for i in range(1, len(self.layers)):
            weights[self.get_n_weight_name(i)] = self.layers[i].weight_matrix
            biases[self.get_n_bias_name(i)] = self.layers[i].bias_vector

        np.savez(weights_file_name, **weights)
        np.savez(bias_file_name, **biases)

    @staticmethod
    def load_config_and_init_predict_network(config_dir:str = "nn_config"):
        config_file_name = config_dir + os.path.sep + "network_args.ini"
        weights_file_name = config_dir + os.path.sep + "weights.npz"
        biases_file_name = config_dir + os.path.sep + "biases.npz"

        if os.path.exists(config_file_name) == False or os.path.exists(weights_file_name) == False or os.path.exists(biases_file_name) == False:
            raise FileNotFoundError("配置文件不全！")

        config = configparser.ConfigParser()
        config.read(config_file_name)
        learn_rate = float(config["main"]["learn_rate"])
        mid_normal_max = int(config["main"]["mid_normal_max"])
        sample_width = int(config["main"]["sample_width"])
        sample_height = int(config["main"]["sample_height"])

        weight_load = np.load(weights_file_name)
        weight_matrix = []
        for i in range(len(weight_load)):
            weight_matrix.append(weight_load[OneHotFullLinkNetwork.get_n_weight_name(i + 1)])

        bias_load = np.load(biases_file_name)
        biases_matrix = []
        for i in range(len(bias_load)):
            biases_matrix.append(bias_load[OneHotFullLinkNetwork.get_n_bias_name(i + 1)])
        
        mid_layer_sizes = [len(x) for x in biases_matrix[:-1]]

        network = OneHotFullLinkNetwork(mid_layer_sizes, "ReLU", learn_rate=learn_rate, mid_normal_max=mid_normal_max, sample_shape=(sample_width, sample_height))
        network.init_network_for_predict(weight_matrix, biases_matrix)

        return network

def train(config_save_dir:str = "nn_config", mid_layer_sizes:list = [160,72], learn_rate:int = 0.05, mid_normal_max = 5, color_reverse:bool = False):
    network = OneHotFullLinkNetwork(mid_layer_sizes, learn_rate=learn_rate, mid_normal_max=mid_normal_max, train_color_reverse=color_reverse)
    # network.load_samples_and_init_network(r"..\source_data\train")
    network.load_samples_and_init_network(r"..\source_data\mnist\train")

    slice_size = 30
    slice_st = 0
    full_sample_size = network.layers[0].full_layer_value.shape[1]
    if slice_size == -1:
        slice_size = full_sample_size

    batch_count = math.ceil(network.layers[0].full_layer_value.shape[1] / slice_size)

    acc_arr = [0 for i in range(batch_count)]
    ensure_arr = [0 for i in range(batch_count)]
    loss_arr = [0 for i in range(batch_count)]

    acc_av = np.average(acc_arr)
    ensure_av = np.average(ensure_arr)
    epoch = 0

    st_time = time.time()
    for i in range(10000000 * batch_count):
        
        if slice_st >= full_sample_size:
            slice_st = 0
        network.forward_broadcasting(slice_st=slice_st, slice_step=slice_size)
        
        acc_arr[i % batch_count] = network.accuracy
        ensure_arr[i % batch_count] = network.ensure_99_precent
        loss_arr[i % batch_count] = network.loss
        acc_av = np.average(acc_arr)
        ensure_av = np.average(ensure_arr)
        loss_av = np.average(loss_arr)
        if acc_av >= 0.999 and ensure_av > 0.9 and loss_av < 0.01:
            break
        if i % batch_count == batch_count - 1:
            print(f"epoch {i // batch_count}, round, acc: {i % batch_count}: {acc_av * 100:.4}%, ensure_99_precent:{ensure_av * 100:.4}%, loss:{loss_av * 100:.4}%")
        network.backward_broadcasting()
        slice_st += slice_size

    test_path = r"..\source_data\test"
    test_path = r"..\source_data\mnist\test"
    test_file_list = os.listdir(test_path)
    correct_count = 0
    end_time = time.time()
    print(f"train cost: {end_time - st_time:.5} s")

    network.save_config(config_save_dir)
    print("Network params saved.")

    st_time = time.time()
    for pure_file_name in test_file_list:
        full_file_name = test_path + os.sep + pure_file_name
        softres = network.predict_test(full_file_name, network.train_color_reverse)
        true_v = int(pure_file_name.split(".")[0])
        pred_v = softres.argmax(axis=0)

        if true_v == pred_v:
            correct_count += 1

        # print(f"For File: {pure_file_name}, Predict: {pred_v}， possibilty:{softres.max() * 100:.4}")

    print(f"Correct_Rate: {correct_count / len(test_file_list) * 100:.4}%")
    end_time = time.time()
    print(f"test cost {end_time - st_time:.5} s")

def use_to_predict(config_dir:str = "nn_config", img_dir = "../test_imgs", color_reverse:bool = False):
    network = OneHotFullLinkNetwork.load_config_and_init_predict_network(config_dir)
    if os.path.exists(img_dir) == False:
        print("Img Dir Not Exists!")
        return

    while True:
        file_name = input("File Name, # to exit:").strip()
        if file_name == "#":
            break

        full_file_name = img_dir + os.path.sep + file_name
        if os.path.exists(full_file_name) == False:
            print(f"img file not exists!")

        img = cv2.imread(full_file_name)
        
        res = network.predict_single_img(img, color_reverse=color_reverse)
        print(f"RESULT: [{res}]")
            
def main():
    choose = input("Train/Predict (Y/N):").strip().lower()

    if choose == "y":
        train("nn_black_words_config",mid_layer_sizes=[192,72])

    else:
        use_to_predict(color_reverse=True)

if __name__ == "__main__":
    main()





        



     
    


    

    


        

