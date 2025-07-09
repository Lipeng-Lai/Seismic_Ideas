import os
import numpy as np
from scipy.signal import hilbert
from tqdm import tqdm


def run(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有 field1_free_*.npy 文件
    npy_files = sorted(f for f in os.listdir(input_dir) if f.endswith('.npy'))

    # 遍历处理
    for fname in tqdm(npy_files, desc="Extracting Hilbert imaginary parts"):
        input_path = os.path.join(input_dir, fname)
        
        # 载入数据
        data = np.load(input_path)
        
        # 希尔伯特变换，并提取虚部
        z = hilbert(data, axis=0)
        imag_part = np.imag(z)  # 只保留虚数部分
        
        # 构造输出文件名（field1 -> field2）
        new_fname = fname.replace('field1_', 'field2_')
        output_path = os.path.join(output_dir, new_fname)
        
        # 保存虚部
        np.save(output_path, imag_part)
    
if __name__ == '__main__':
    input_dir = '../dataset/multiple'
    output_dir = '../dataset/multiple_hilbert'
    run(input_dir, output_dir)
    
    input_dir = '../dataset/image'
    output_dir = '../dataset/image_hilbert'
    run(input_dir, output_dir)
