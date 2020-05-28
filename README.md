# Facial_expression_recognition_DeepCNN
Using VGG or Resnet to learn facial expression recognition with FER2013 dataset

## Training and Original Results



## Prune Model (Network Slimming)



### Introduction


Network Slimming is a neural network training scheme that can simultaneously reduce the model size, run-time memory, computing operations, while introducing no accuracy loss to and minimum overhead to the training process. The resulting models require no special libraries/hardware for efficient inference.


## Example Usage
  
This repo holds the example code for VGGNet on CIFAR-10 dataset. 

0. Prepare the directories to save the results

1. Train vgg network with channel level sparsity, S is the lambda in the paper which controls the significance of sparsity

```
th main.lua -netType vgg -save vgg_cifar10/ -S 0.0001
```
 2. Identify a certain percentage of relatively unimportant channels and set their scaling factors to 0

```
th prune/prune.lua -percent 0.7 -model vgg_cifar10/model_160.t7  -save vgg_cifar10/pruned/model_160_0.7.t7
```
 3. Re-build a real compact network and copy the weights from the model in the last stage

```
th convert/vgg.lua -model vgg_cifar10/pruned/model_160_0.7.t7 -save vgg_cifar10/converted/model_160_0.7.t7
```
 4. Fine-tune the compact network
 
```
th main_fine_tune.lua -retrain vgg_cifar10/converted/model_160_0.7.t7 -save vgg_cifar10/fine_tune/
```
## Reference
* [Learning Efficient Convolutional Networks through Network Slimming](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf) (ICCV 2017).

* [Zhuang Liu](https://liuzhuang13.github.io/), [Jianguo Li](https://sites.google.com/site/leeplus/), [Zhiqiang Shen](http://zhiqiangshen.com), [Gao Huang](http://www.cs.cornell.edu/~gaohuang/), [Shoumeng Yan](https://scholar.google.com/citations?user=f0BtDUQAAAAJ&hl=en), [Changshui Zhang](http://bigeye.au.tsinghua.edu.cn/english/Introduction.html).

* Liu, Zhuang and Li, Jianguo and Shen, Zhiqiang and Huang, Gao and Yan, Shoumeng and Zhang, Changshui, Learning Efficient Convolutional Networks through Network Slimming，ICCV 2017



