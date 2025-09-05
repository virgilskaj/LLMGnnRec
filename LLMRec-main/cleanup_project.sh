#!/bin/bash

echo "🗑️ LLMRec项目文件清理脚本"
echo "🎯 保留成功配置所需的核心文件"
echo "=" * 50

# 显示当前项目大小
echo "📊 清理前项目大小:"
du -sh . 2>/dev/null || echo "无法计算大小"

echo ""
echo "🗂️ 开始清理无关文件..."

# 删除实验性模型文件
echo "🔧 删除实验性模型文件..."
rm -f Models_Enhanced.py
rm -f Models_Simple_Enhanced.py  
rm -f Models_Conservative_Enhanced.py
rm -f Models_Optimized_Enhanced.py
rm -f Models_Final_Optimized.py
rm -f Models_Exact_Enhanced.py

# 删除实验性训练脚本
echo "📝 删除实验性训练脚本..."
rm -f main_enhanced.py
rm -f main_simple_enhanced.py
rm -f main_optimized_enhanced.py

# 删除调试和测试脚本
echo "🧪 删除调试测试脚本..."
rm -f test_enhanced_model.py
rm -f validate_baseline.py
rm -f final_performance_test.py
rm -f quick_performance_test.py
rm -f tune_with_fixed_lr.py
rm -f quick_param_test.py
rm -f improve_weak_metrics.py
rm -f hyperparameter_tuning.py

# 删除运行脚本
echo "🏃 删除运行脚本..."
rm -f run_comparison.py
rm -f run_fixed_enhanced.py
rm -f run_performance_boost.py

# 删除文档文件 (保留核心README)
echo "📚 删除实验文档..."
rm -f ENHANCED_README.md
rm -f INTEGRATION_SUMMARY.md
rm -f QUICK_FIX_GUIDE.md
rm -f PERFORMANCE_BOOST_GUIDE.md
rm -f FINAL_SOLUTION.md
rm -f SUCCESS_REPORT.md
rm -f targeted_tests.md
rm -f FILE_CLEANUP_GUIDE.md

# 删除缓存和临时文件
echo "🧹 删除缓存文件..."
rm -rf __pycache__/
rm -rf venv/ 2>/dev/null

# 删除重复文件
echo "📄 删除重复文件..."
rm -f "requirements - 副本.txt"

# 可选：删除未使用的目录 (谨慎删除，可能包含有用数据)
echo ""
echo "⚠️  可选删除目录 (请手动确认):"
echo "   - MMSSL/                     # 如果不使用MMSSL基线"
echo "   - LLM_augmentation_construct_prompt/  # 如果数据已预处理完成"
echo ""
echo "   如需删除，请手动执行:"
echo "   rm -rf MMSSL/"
echo "   rm -rf LLM_augmentation_construct_prompt/"

# 显示清理后项目大小
echo ""
echo "📊 清理后项目大小:"
du -sh . 2>/dev/null || echo "无法计算大小"

echo ""
echo "✅ 文件清理完成!"
echo ""
echo "🎯 保留的核心文件可以运行您的成功配置:"
echo "   python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 --embed_size 80 --model_cat_rate 0.03"
echo ""
echo "📁 核心文件结构:"
echo "   main.py                  # 主训练脚本"
echo "   Models.py                # 原始模型"
echo "   Models_EmerG_Lite.py     # 您成功的增强模型"
echo "   utility/                 # 核心工具包"
echo "   data/                    # 数据目录"
echo ""
echo "💡 如需恢复实验文件，请从备份中还原。"