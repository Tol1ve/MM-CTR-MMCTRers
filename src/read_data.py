import pandas as pd

# 定义文件路径
file_path = '/home/chenruiyue/competition/WWW2025_MMCTR/WWW2025_MMCTR_Challenge/data/MicroLens_1M_x1/train.parquet'

try:
    # 读取 Parquet 文件
    df = pd.read_parquet(file_path)
    print('数据读取成功！')
    # 查看数据的基本信息
    print('数据基本信息：')
    df.info()
    # 查看数据集行数和列数
    rows, columns = df.shape

    if rows < 5:
        # 行数少于 5 则查看全量数据信息
        print('数据全部内容信息：')
        print(df.to_csv(sep='\t', na_rep='nan'))
    else:
        # 查看数据前几行信息
        print('数据前几行内容信息：')
        print(df.head().to_csv(sep='\t', na_rep='nan'))

except FileNotFoundError:
    print(f'文件 {file_path} 未找到，请检查文件路径是否正确。')
except Exception as e:
    print(f'读取文件时出现错误：{e}')