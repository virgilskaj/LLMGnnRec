# 🎯 针对性参数调优测试

基于您成功的配置 `--lr 0.0002 --layers 2 --use_enhanced_gnn True`，以下是针对弱指标优化的具体测试命令：

## 🔧 嵌入维度调优 (影响表达能力)

```bash
# 测试1: 较小嵌入维度 (可能提高R@10精度)
python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 --embed_size 80 --title "Test_Embed80"

# 测试2: 中等嵌入维度
python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 --embed_size 96 --title "Test_Embed96"

# 测试3: 更大嵌入维度
python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 --embed_size 160 --title "Test_Embed160"
```

## 🎨 融合权重调优 (最关键参数)

```bash
# 测试4: 降低模型融合权重
python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 --model_cat_rate 0.015 --title "Test_ModelCat015"

# 测试5: 提升模型融合权重  
python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 --model_cat_rate 0.035 --title "Test_ModelCat035"

# 测试6: 调整用户权重
python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 --user_cat_rate 3.5 --title "Test_UserCat35"

# 测试7: 提升用户权重
python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 --user_cat_rate 4.5 --title "Test_UserCat45"
```

## 🎛️ 正则化调优 (影响泛化性能)

```bash
# 测试8: 轻度dropout
python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 --drop_rate 0.02 --title "Test_Drop002"

# 测试9: 中度dropout
python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 --drop_rate 0.05 --title "Test_Drop005"

# 测试10: 调整权重衰减
python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 --weight_decay 0.0005 --title "Test_WeightDecay"
```

## 📦 批次大小调优 (影响梯度质量)

```bash
# 测试11: 小批次 (更精细的梯度)
python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 --batch_size 512 --title "Test_Batch512"

# 测试12: 中等批次
python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 --batch_size 768 --title "Test_Batch768"

# 测试13: 大批次
python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 --batch_size 2048 --title "Test_Batch2048"
```

## 🎯 组合优化 (综合调优)

```bash
# 测试14: Top-10专用配置
python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 \
    --embed_size 80 --model_cat_rate 0.03 --user_cat_rate 4.0 --drop_rate 0.03 --title "Test_Top10_Focused"

# 测试15: NDCG专用配置
python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 \
    --embed_size 96 --model_cat_rate 0.04 --user_cat_rate 4.5 --item_cat_rate 0.01 --title "Test_NDCG_Focused"

# 测试16: 平衡优化配置
python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 \
    --embed_size 96 --model_cat_rate 0.035 --user_cat_rate 3.8 --drop_rate 0.02 --batch_size 768 --title "Test_Balanced"

# 测试17: 激进配置
python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 \
    --embed_size 128 --model_cat_rate 0.06 --user_cat_rate 5.0 --item_cat_rate 0.015 --title "Test_Aggressive"
```

## 📊 关键参数影响分析

### `embed_size` (嵌入维度)
- **80-96**: 可能提高R@10精度，减少过拟合
- **128-160**: 更强表达能力，但可能需要更多数据

### `model_cat_rate` (模型融合权重) 
- **0.015-0.025**: 保守融合，可能改进精确度
- **0.035-0.045**: 中等融合，平衡性能
- **0.05-0.06**: 激进融合，最大化多模态效果

### `user_cat_rate` (用户特征权重)
- **3.5-4.0**: 适中的用户建模
- **4.5-5.0**: 强化用户特征，可能改进个性化

### `drop_rate` (Dropout率)
- **0.02-0.05**: 轻度正则化，改进泛化
- **0.08-0.1**: 强正则化，防止过拟合

## 💡 推荐测试顺序

**第一优先级** (最可能改进弱指标):
```bash
python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 --embed_size 80 --model_cat_rate 0.03 --title "Priority1"
```

**第二优先级**:
```bash
python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 --embed_size 96 --model_cat_rate 0.035 --user_cat_rate 3.8 --title "Priority2"
```

**第三优先级**:
```bash
python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 --embed_size 96 --model_cat_rate 0.04 --drop_rate 0.03 --batch_size 768 --title "Priority3"
```

## 🔄 自动化测试

如果想要自动运行所有测试:
```bash
# 运行完整的参数调优
python3 tune_with_fixed_lr.py

# 或者运行快速测试
python3 quick_param_test.py
```

---

**建议**: 先手动运行前3个优先级测试，观察哪个方向有改进，然后在该方向上进一步细化！🎯