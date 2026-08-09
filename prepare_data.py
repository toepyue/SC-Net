import os
import random
import shutil

def prepare_and_split_aid_dataset(source_base_dir, target_base_dir):
    print("开始执行数据集扁平化与严格划分任务...")
    
    # 论文中明确要求的比例与数量[cite: 1]
    num_train = 9000
    num_val = 500
    num_test = 500
    
    # 确保目标文件夹存在
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(target_base_dir, split), exist_ok=True)
        
    # 1. 递归获取 source_base_dir 下所有的图片文件（扁平化读取）
    all_images = []
    for root, _, files in os.walk(source_base_dir):
        for file in files:
            # 过滤提取图片格式
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, file))
                
    total_images = len(all_images)
    print(f"在原始数据集中共找到 {total_images} 张图像。")
    
    if total_images < (num_train + num_val + num_test):
        print(f"警告：源文件夹中的图像数量 ({total_images}) 不足 10000 张！请检查数据集是否完整解压。")
        return
        
    # 设定随机种子以保证每次划分结果的一致性
    random.seed(42)
    # 打乱数据集，确保不同地貌特征均匀分布到各个集中
    random.shuffle(all_images)
    
    # 划分列表[cite: 1]
    train_images = all_images[:num_train]
    val_images = all_images[num_train : num_train+num_val]
    test_images = all_images[num_train+num_val : num_train+num_val+num_test]
    
    # 定义移动文件的辅助函数
    def copy_files(file_path_list, split_name):
        print(f"正在拷贝 {split_name} 集 ({len(file_path_list)} 张)...")
        for idx, src_path in enumerate(file_path_list):
            # 2. 为防止不同子文件夹下有同名文件，用序号给新文件统一重新命名
            ext = os.path.splitext(src_path)[1]
            new_file_name = f"{split_name}_{idx+1:05d}{ext}"
            dst_path = os.path.join(target_base_dir, split_name, new_file_name)
            
            # 使用 copy 而不是 move，方便出错时重来，不破坏原始数据
            shutil.copy(src_path, dst_path) 
        print(f"{split_name} 集已成功划分并拷贝完毕！")

    # 执行拷贝任务
    copy_files(train_images, 'train')
    copy_files(val_images, 'val')
    copy_files(test_images, 'test')
    
    print("\n--- 9000/500/500 数据集扁平化与划分全部成功闭环！---")

if __name__ == "__main__":
    # AID 数据集解压在此目录（无论里面嵌套了多少层子文件夹）
    raw_source = "./raw_data" 
    # 我们将其清洗并打平划分到 ./data 目录下
    target_dir = "./data"
    
    if os.path.exists(raw_source):
        prepare_and_split_aid_dataset(raw_source, target_dir)
    else:
        print("尚未找到 raw_data 文件夹，请确保数据已经上传并解压！")
        