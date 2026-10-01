"""
FScanpy数据模块
提供测试数据访问和处理功能
"""

import os
from pathlib import Path
from typing import List

def get_test_data_path(filename: str) -> str:
    """
    获取测试数据文件的完整路径
    
    Args:
        filename: 测试数据文件名
        
    Returns:
        str: 文件的完整路径
        
    Examples:
        >>> from FScanpy.data import get_test_data_path
        >>> blastx_file = get_test_data_path('blastx_example.xlsx')
        >>> mrna_file = get_test_data_path('mrna_example.fasta')
        >>> region_file = get_test_data_path('region_example.csv')
        >>> full_seq_file = get_test_data_path('full_seq.xlsx')
    """
    current_dir = Path(__file__).parent
    test_data_dir = current_dir / "test_data"
    file_path = test_data_dir / filename
    
    if not file_path.exists():
        available_files = list_test_data()
        raise FileNotFoundError(
            f"测试数据文件不存在: {filename}\n"
            f"可用的测试数据文件: {available_files}"
        )
    
    return str(file_path)

def list_test_data() -> List[str]:
    """
    列出所有可用的测试数据文件
    
    Returns:
        List[str]: 测试数据文件名列表
        
    Examples:
        >>> from FScanpy.data import list_test_data
        >>> files = list_test_data()
        >>> print(files)
        ['blastx_example.xlsx', 'mrna_example.fasta', 'region_example.csv']
    """
    try:
        current_dir = Path(__file__).parent
        test_data_dir = current_dir / "test_data"
        
        if not test_data_dir.exists():
            return []
        
        files = []
        for file_path in test_data_dir.iterdir():
            if file_path.is_file() and not file_path.name.startswith('.'):
                files.append(file_path.name)
        
        return sorted(files)
        
    except Exception:
        return []

def print_test_data_info():
    """
    打印测试数据的详细信息
    """
    print("📋 FScanpy 测试数据信息:")
    print("=" * 50)
    
    try:
        current_dir = Path(__file__).parent
        test_data_dir = current_dir / "test_data"
        
        if not test_data_dir.exists():
            print("❌ 测试数据目录不存在")
            return
        
        files = list_test_data()
        if not files:
            print("❌ 没有找到测试数据文件")
            return
        
        print(f"📁 数据目录: {test_data_dir}")
        print(f"📊 文件数量: {len(files)}")
        print()
        
        file_descriptions = {
            'blastx_example.xlsx': '🧬 BLASTX比对结果示例 (1000条记录)',
            'mrna_example.fasta': '🧬 mRNA序列示例数据',
            'region_example.csv': '🎯 PRF区域验证数据 (含标签)',
            'full_seq.xlsx': '🧬 完整序列示例数据'
        }
        
        for filename in files:
            file_path = test_data_dir / filename
            size_mb = file_path.stat().st_size / (1024 * 1024)
            
            description = file_descriptions.get(filename, '📄 数据文件')
            print(f"  {description}")
            print(f"    文件名: {filename}")
            print(f"    大小: {size_mb:.2f} MB")
            print(f"    路径: {file_path}")
            print()
        
        print("🚀 使用示例:")
        print("  from FScanpy.data import get_test_data_path")
        print("  blastx_file = get_test_data_path('blastx_example.xlsx')")
        print("  mrna_file = get_test_data_path('mrna_example.fasta')")
        print("  region_file = get_test_data_path('region_example.csv')")
        print("  full_seq_file = get_test_data_path('full_seq.xlsx')")
    except Exception as e:
        print(f"❌ 获取数据信息时出错: {e}")

# 导出主要函数
__all__ = ['get_test_data_path', 'list_test_data', 'print_test_data_info']