#!/bin/bash

echo "🗑️ GitHub仓库清理脚本"
echo "🎯 删除冗余代码，保留成功配置的核心文件"
echo "=" * 60

# 检查当前分支
current_branch=$(git branch --show-current)
echo "📍 当前分支: $current_branch"

# 确认是否继续
echo ""
echo "⚠️  警告: 此脚本将从Git仓库中删除以下文件:"
echo "   - 实验性模型文件 (6个)"
echo "   - 实验性训练脚本 (3个)"  
echo "   - 调试测试脚本 (8个)"
echo "   - 运行脚本 (3个)"
echo "   - 文档文件 (8个)"
echo "   - 缓存和临时文件"
echo ""
echo "✅ 将保留您成功配置需要的核心文件:"
echo "   - main.py (已修改支持增强)"
echo "   - Models.py (原始模型)"
echo "   - Models_EmerG_Lite.py (成功的增强模型)"
echo "   - utility/ (核心工具)"
echo "   - data/ (数据目录)"
echo "   - requirements.txt"
echo "   - README.md"
echo ""

read -p "🤔 确认继续清理? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 清理已取消"
    exit 1
fi

echo "🚀 开始清理GitHub仓库..."

# 1. 删除实验性模型文件
echo "🔧 删除实验性模型文件..."
git rm -f Models_Enhanced.py 2>/dev/null || echo "   Models_Enhanced.py 不存在"
git rm -f Models_Simple_Enhanced.py 2>/dev/null || echo "   Models_Simple_Enhanced.py 不存在"
git rm -f Models_Conservative_Enhanced.py 2>/dev/null || echo "   Models_Conservative_Enhanced.py 不存在"
git rm -f Models_Optimized_Enhanced.py 2>/dev/null || echo "   Models_Optimized_Enhanced.py 不存在"
git rm -f Models_Final_Optimized.py 2>/dev/null || echo "   Models_Final_Optimized.py 不存在"
git rm -f Models_Exact_Enhanced.py 2>/dev/null || echo "   Models_Exact_Enhanced.py 不存在"

# 2. 删除实验性训练脚本
echo "📝 删除实验性训练脚本..."
git rm -f main_enhanced.py 2>/dev/null || echo "   main_enhanced.py 不存在"
git rm -f main_simple_enhanced.py 2>/dev/null || echo "   main_simple_enhanced.py 不存在"
git rm -f main_optimized_enhanced.py 2>/dev/null || echo "   main_optimized_enhanced.py 不存在"

# 3. 删除调试测试脚本
echo "🧪 删除调试测试脚本..."
git rm -f test_enhanced_model.py 2>/dev/null || echo "   test_enhanced_model.py 不存在"
git rm -f validate_baseline.py 2>/dev/null || echo "   validate_baseline.py 不存在"
git rm -f final_performance_test.py 2>/dev/null || echo "   final_performance_test.py 不存在"
git rm -f quick_performance_test.py 2>/dev/null || echo "   quick_performance_test.py 不存在"
git rm -f tune_with_fixed_lr.py 2>/dev/null || echo "   tune_with_fixed_lr.py 不存在"
git rm -f quick_param_test.py 2>/dev/null || echo "   quick_param_test.py 不存在"
git rm -f improve_weak_metrics.py 2>/dev/null || echo "   improve_weak_metrics.py 不存在"
git rm -f hyperparameter_tuning.py 2>/dev/null || echo "   hyperparameter_tuning.py 不存在"

# 4. 删除运行脚本
echo "🏃 删除运行脚本..."
git rm -f run_comparison.py 2>/dev/null || echo "   run_comparison.py 不存在"
git rm -f run_fixed_enhanced.py 2>/dev/null || echo "   run_fixed_enhanced.py 不存在"
git rm -f run_performance_boost.py 2>/dev/null || echo "   run_performance_boost.py 不存在"

# 5. 删除文档文件
echo "📚 删除实验文档..."
git rm -f ENHANCED_README.md 2>/dev/null || echo "   ENHANCED_README.md 不存在"
git rm -f INTEGRATION_SUMMARY.md 2>/dev/null || echo "   INTEGRATION_SUMMARY.md 不存在"
git rm -f QUICK_FIX_GUIDE.md 2>/dev/null || echo "   QUICK_FIX_GUIDE.md 不存在"
git rm -f PERFORMANCE_BOOST_GUIDE.md 2>/dev/null || echo "   PERFORMANCE_BOOST_GUIDE.md 不存在"
git rm -f FINAL_SOLUTION.md 2>/dev/null || echo "   FINAL_SOLUTION.md 不存在"
git rm -f SUCCESS_REPORT.md 2>/dev/null || echo "   SUCCESS_REPORT.md 不存在"
git rm -f targeted_tests.md 2>/dev/null || echo "   targeted_tests.md 不存在"
git rm -f FILE_CLEANUP_GUIDE.md 2>/dev/null || echo "   FILE_CLEANUP_GUIDE.md 不存在"

# 6. 删除清理相关文件
echo "🧹 删除清理工具..."
git rm -f DELETE_THESE_FILES.txt 2>/dev/null || echo "   DELETE_THESE_FILES.txt 不存在"
git rm -f cleanup_project.sh 2>/dev/null || echo "   cleanup_project.sh 不存在"

# 7. 删除重复文件
echo "📄 删除重复文件..."
git rm -f "requirements - 副本.txt" 2>/dev/null || echo "   requirements - 副本.txt 不存在"

# 检查是否有文件被删除
deleted_files=$(git status --porcelain | grep "^D" | wc -l)

if [ $deleted_files -gt 0 ]; then
    echo ""
    echo "📋 准备删除的文件:"
    git status --porcelain | grep "^D" | sed 's/^D  /   - /'
    
    echo ""
    echo "📊 删除文件统计: $deleted_files 个文件"
    
    # 提交删除
    echo ""
    echo "💾 提交删除到Git..."
    git commit -m "🗑️ Clean up redundant experimental files

✅ 保留成功配置的核心文件:
- main.py (支持EmerG增强)
- Models_EmerG_Lite.py (成功的增强模型)
- Models.py (原始模型)
- utility/ (核心工具)

🎯 成功配置: --use_enhanced_gnn True --layers 2 --lr 0.0002 --embed_size 80 --model_cat_rate 0.03

📊 性能提升: 所有指标均超越原论文结果

🗑️ 删除了 $deleted_files 个实验性和调试文件"

    echo ""
    echo "🚀 推送到GitHub..."
    git push origin $current_branch
    
    if [ $? -eq 0 ]; then
        echo "✅ 成功推送到GitHub!"
        echo ""
        echo "🎯 GitHub仓库清理完成:"
        echo "   - 删除了 $deleted_files 个冗余文件"
        echo "   - 保留了您成功配置的核心文件"
        echo "   - 仓库大小显著减少"
        echo ""
        echo "🔗 您的成功配置仍然可用:"
        echo "   python3 main.py --dataset netflix --epoch 200 --use_enhanced_gnn True --layers 2 --lr 0.0002 --embed_size 80 --model_cat_rate 0.03"
    else
        echo "❌ 推送失败，请检查网络连接或权限"
        echo "💡 您可以稍后手动推送: git push origin $current_branch"
    fi
    
else
    echo ""
    echo "ℹ️  没有找到需要删除的文件 (可能已经被本地清理)"
    echo "📊 当前仓库状态:"
    git status --short
fi

echo ""
echo "🎉 GitHub清理脚本执行完成!"