## anchor 生成源码解读

涉及源码：ssd_head.py和anchor_generator.py
---------------------
anchor基本介绍: anchor设计和caffe ssd anchor设计一致, 假设min_size为a, max_size为b, 则先生成ratio为1, 宽度和高度为(a, a), 
(sqrt(ab), sqrt(ab))的两个anchor, ratio为2, 1/2, 3, 1/3则分别生成宽度和高度为(a*sqrt(ratio), a/sqrt(ratio))的anchor, 
mmdetection中必须设定每一层的min_size, max_size, 因此ratios为[2]则对应4个anchor, ratios为[2,3]则对应6个anchor

### ssd_head.py涉及anchor部分
在init()函数中,先生成min_size, max_size, 注意它这里是必须要指定max_size(和caffe SSD不同,无法生成奇数个anchor), 确保len(min_size)=len(max_size), 
调用AnchorGenerator()类生成了base_anchors, 数量是6或者10,使用indices操作从6个anchor里选择(0, 3, 1, 2)或者从10个anchor里选择(0, 5, 1, 2, 3, 4)
-> 最终生成4个或者6个anchor

由于在多个feature map上生成anchor,因此使用了一个for循环操作, 将anchor_generator放入到anchor_generatos[]中

### anchor_generator.py
提供了一个AnchorGenerator类, init()函数需要如下参数:
* base_size: 即设置的min_size 
* scales: 是(1, sqrt(max_size / min_size)), 用来生成ratio为1的两个anchor
* ratios: 是(1, 2, 1/2)或者(1, 2, 1/2, 3, 1/3)
* ctr: ctr由stride生成, 是anchor的中心坐标, ((stride - 1) /2, (stride - 1) / 2) 
在gen_base_anchor()函数里, 使用上面的参数来计算base_anchor, 计算流程如下:
* 根据ratios来计算h_ratios和w_ratios, 即上面所述的(1 / sqrt(ratios), sqrt(ratios)) (note: 此处顺序反了,不影响结果)
* 根据scales来计算base_size, 一共有2个分别是(min_size, sqrt(min_size * max_size)) = min_size * scales
* 计算anchors的宽度和高度, 只以宽度举例: w = base_size * w_ratios, 以ratios是(1, 2, 1/2)举例, base_size shape为(2, 1), 
w_ratios shape为(1, 3), 计算出的w是(2, 3) **一共生成了6个anchor, 如果ratios是(1, 2, 1/2, 3, 1/3), 则生成10个anchor
(此处anchor数量和标准ssd anchor数量不一致 -> 再筛选(即ssd_head.py中使用indices操作进行筛选))**
