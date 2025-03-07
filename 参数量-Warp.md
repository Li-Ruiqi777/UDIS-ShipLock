以下FLOPs都是在batch size为1的情况下计算得到的
计算速度与FLOPs强相关, 与Params的数量关系不大
UDIS2多的参数量应该在回归网络中的全连接层

1.UDIS2
```
Number of Params: 78.00 M
Number of FLOPs: 35.75 G
```

2.特征提取用ResNet50 + MeshRegress中间层调小 + HmomoRegress自己写的
```
Number of Params: 18.55 M
Number of FLOPs: 35.60 G
```

3.特征提取用ResNet50 + MeshRegress中间层调小+ HmomoRegress用的跟原始论文一样
```
Number of Params: 37.53 M
Number of FLOPs: 35.70 G
```

4.ResNet50 + 2个回归网络改成残差连接 + 使用全局最大池化再全连接
```
Number of Params: 36.23 M
Number of FLOPs: 35.32 G
```

5.YOLO11-l的BackBone + 2个回归网络改成残差连接 + 使用SPP(空间金字塔池化)再全连接
```
Number of Params: 84.13 M
Number of FLOPs: 20.84 G
```

6.YOLO11-n的BackBone + FPN + 2个回归网络改成残差连接 + 使用SPP(空间金字塔池化)再全连接
```
Number of Params: 81.78 M
Number of FLOPs: 4.23 G
```

7.UDIS2替换BacKBone为MobileNet-V4
```
Number of Params: 77.94 M
Number of FLOPs: 10.88 G
```

8.UDIS2替换BacKBone为MobileNet-V4 + AFPN
```
Number of Params: 83.46 M
Number of FLOPs: 14.75 G
```

9.UDIS2 + FPN
```
Number of Params: 79.54 M
Number of FLOPs: 42.95 G
```

10.主干换成YOLO11(width=0.75, depth=1.0)
```
Number of Params: 72.85 M
Number of FLOPs: 16.47 G
```