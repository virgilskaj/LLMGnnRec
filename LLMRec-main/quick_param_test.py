#!/usr/bin/env python3
"""
Quick parameter test with fixed lr=0.0002
Focus on the most impactful parameters for weak metrics improvement
"""

import os
import sys

def run_single_test(params_str, test_name):
    """Run a single test configuration"""
    base_cmd = "python3 main.py --dataset netflix --epoch 100 --use_enhanced_gnn True --layers 2 --lr 0.0002"
    full_cmd = f"{base_cmd} {params_str} --title {test_name}"
    
    print(f"\n🧪 {test_name}")
    print(f"🔧 参数: {params_str}")
    print(f"📋 运行: {full_cmd}")
    print("-" * 50)
    
    exit_code = os.system(full_cmd)
    
    if exit_code == 0:
        print(f"✅ {test_name} 完成")
    else:
        print(f"❌ {test_name} 失败")
    
    return exit_code == 0

def main():
    print("⚡ 快速参数测试 - 固定学习率 0.0002")
    print("=" * 60)
    print("🎯 目标: 改进弱指标 R@10, N@10, N@20")
    print("📊 基线: R@20=0.08618, R@50=0.14797")
    print("=" * 60)
    
    # Key parameter tests designed to improve weak metrics
    test_configs = [
        # Test 1: 调整嵌入维度 (可能影响表达能力)
        ("--embed_size 96", "Test1_Embed96"),
        ("--embed_size 80", "Test2_Embed80"),
        
        # Test 2: 调整模型融合权重 (最关键的参数)
        ("--model_cat_rate 0.035", "Test3_ModelCat035"),
        ("--model_cat_rate 0.025", "Test4_ModelCat025"),
        ("--model_cat_rate 0.045", "Test5_ModelCat045"),
        
        # Test 3: 调整用户融合权重
        ("--user_cat_rate 3.5", "Test6_UserCat35"),
        ("--user_cat_rate 4.5", "Test7_UserCat45"),
        
        # Test 4: 调整物品属性权重
        ("--item_cat_rate 0.008", "Test8_ItemCat008"),
        ("--item_cat_rate 0.012", "Test9_ItemCat012"),
        
        # Test 5: 调整dropout (可能影响泛化)
        ("--drop_rate 0.02", "Test10_Drop002"),
        ("--drop_rate 0.08", "Test11_Drop008"),
        
        # Test 6: 调整批次大小
        ("--batch_size 768", "Test12_Batch768"),
        ("--batch_size 1536", "Test13_Batch1536"),
        
        # Test 7: 组合优化 (基于理论分析)
        ("--embed_size 96 --model_cat_rate 0.035 --user_cat_rate 3.8", "Test14_Combo1"),
        ("--embed_size 80 --model_cat_rate 0.04 --user_cat_rate 4.2 --drop_rate 0.03", "Test15_Combo2"),
        ("--embed_size 128 --model_cat_rate 0.03 --user_cat_rate 3.5 --item_cat_rate 0.01", "Test16_Combo3"),
        
        # Test 8: 激进配置 (更大胆的尝试)
        ("--embed_size 160 --model_cat_rate 0.06 --user_cat_rate 5.0", "Test17_Aggressive"),
        ("--embed_size 64 --model_cat_rate 0.08 --user_cat_rate 6.0", "Test18_HighFusion")
    ]
    
    print(f"🚀 将运行 {len(test_configs)} 个测试配置...")
    
    successful_tests = 0
    failed_tests = 0
    
    for params_str, test_name in test_configs:
        success = run_single_test(params_str, test_name)
        if success:
            successful_tests += 1
        else:
            failed_tests += 1
        
        print(f"\n📊 进度: {successful_tests + failed_tests}/{len(test_configs)} 完成")
    
    print("\n" + "=" * 60)
    print("📋 测试完成总结")
    print("=" * 60)
    print(f"✅ 成功: {successful_tests}")
    print(f"❌ 失败: {failed_tests}")
    
    print("\n💡 下一步:")
    print("   1. 查看各个测试的日志文件，找到最佳性能")
    print("   2. 选择表现最好的配置运行完整训练 (--epoch 300)")
    print("   3. 如果仍需改进，可以尝试:")
    print("      - 更长的训练时间")
    print("      - 学习率调度")
    print("      - 高级增强模型")

if __name__ == '__main__':
    main()