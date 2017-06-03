# alphaPLM
## 前言：
* alphaPLM是Large Scale Piece-wise Linear Model(LS-PLM)的一个单机多线程版本实现，用于解决二分类问题，比如CTR预估，优化算法采用了FTRL。<br>

* LS-PLM据说是之前阿里广告主要的ctr预估模型，早期应该叫做MLR，具体算法原理见盖坤大神最近放出的论文: Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction，代码实现所用优化方法见我的博客文章：http://castellanzhang.github.io/2017/06/01/mlr_plm/<br>


* 此代码基于之前的[alphaFM](https://github.com/CastellanZhang/alphaFM)修改实现，且正赶上alphaGo再次完虐人类，便仍然冠以alpha前缀。<br>

* 安装方法和使用方法跟[alphaFM](https://github.com/CastellanZhang/alphaFM)非常类似，不再详述。<br>

## 模型文件格式（假定分片数为f）：
第一行是bias的参数：<br>
`bias u1 u2 ... uf w1 w2 ... wf u_n1 u_n2 ... u_nf w_n1 w_n2 ... w_nf u_z1 u_z2 ... u_zf w_z1 w_z2 ... w_zf`<br>
其他行的格式为：<br>
`feature_name u1 u2 ... uf w1 w2 ... wf u_n1 u_n2 ... u_nf w_n1 w_n2 ... w_nf u_z1 u_z2 ... u_zf w_z1 w_z2 ... w_zf`<br>
## 预测结果格式：
`label score`<br>
其中label为1或-1，score等于预测为正样本的概率值。<br>

## 参数说明（可以直接执行./plm_train和./plm_predict查看参数列表）：
### plm_train的参数：
-m: 设置模型文件的输出路径。<br>
-u_bias: 1或0表示u是否有偏置项。	default:1<br>
-w_bias: 1或0表示w是否有偏置项。	default:1<br>
-piece_num: 分片数。	default:4<br>
-u_stdev: u的初始化使用均值为0的高斯分布，u_stdev为标准差。	default:0.1<br>
-w_stdev: w的初始化使用均值为0的高斯分布，w_stdev为标准差。	default:0.1<br>
-u_alpha: u的FTRL超参数alpha。	default:0.05<br>
-u_beta: u的FTRL超参数beta。	default:1.0<br>
-u_l1: u的L1正则。	default:0.1<br>
-u_l2: u的L2正则。	default:5.0<br>
-w_alpha: w的FTRL超参数alpha。	default:0.05<br>
-w_beta: w的FTRL超参数beta。	default:1.0<br>
-w_l1: w的L1正则。	default:0.1<br>
-w_l2: w的L2正则。	default:5.0<br>
-core: 计算线程数。	default:1<br>
-im: 上次模型的路径，用于初始化模型参数。如果是第一次训练则不用设置此参数。<br>
### plm_predict的参数：
-m: 模型文件路径。<br>
-piece_num: 分片数。	default:4<br>
-core: 计算线程数。	default:1<br>
-out: 输出文件路径。<br>
