import os
import numpy as np
import logging
from tqdm import tqdm
from funasr import AutoModel

def extract_and_save_embeddings(input_dir, output_dir, model_name="iic/emotion2vec_plus_large"):
    """
    遍历 input_dir 中的所有 .wav 文件（包括子目录），提取情感嵌入，并将其保存为 .npy 文件到 output_dir。
    保存的 .npy 文件名与原始 .wav 文件名相同。

    :param input_dir: 包含音频文件的输入目录路径
    :param output_dir: 保存 .npy 嵌入的输出目录路径（所有文件放在一个文件夹中）
    :param model_name: 使用的模型名称，默认为 "iic/emotion2vec_base"
    """

    # 配置日志，只记录 INFO 及以上级别的信息
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("embedding_extraction.log"),
            logging.StreamHandler()
        ]
    )

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 初始化模型
    try:
        model = AutoModel(model=model_name)
        logging.info(f"模型 '{model_name}' 初始化成功。")
    except Exception as e:
        logging.error(f"初始化模型 '{model_name}' 失败: {e}")
        return

    # 获取输入目录的绝对路径
    input_dir_abs = os.path.abspath(input_dir)

    # 收集所有 .wav 文件
    wav_files = []
    for root, _, files in os.walk(input_dir_abs):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    total_files = len(wav_files)
    logging.info(f"找到 {total_files} 个 .wav 文件。开始处理...")

    # 使用 tqdm 添加进度条
    for wav_path in tqdm(wav_files, desc="Processing .wav files", unit="file"):
        try:
            # 提取情感嵌入
            res = model.generate(
                wav_path,
                output_dir=None,
                granularity="utterance",
                extract_embedding=True
            )
            embedding = None
            if isinstance(res, dict):
                embedding = res.get('feats')
                if embedding is None:
                    logging.warning(f"'feats' not found in the dictionary for {wav_path}")
            elif isinstance(res, list):
                if len(res) > 0:
                    first_item = res[0]
                    if isinstance(first_item, dict) and 'feats' in first_item:
                        embedding = first_item['feats']
                    elif isinstance(first_item, (list, np.ndarray)):
                        embedding = first_item
                    else:
                        logging.warning(f"Unexpected structure in list for {wav_path}: {first_item}")
            else:
                logging.warning(f"Unexpected type for res: {type(res)}")

            if embedding is not None:
                try:
                    embedding = np.array(embedding, dtype=np.float32)
                except Exception as e:
                    logging.error(f"无法将嵌入向量转换为 NumPy 数组: {e}")
                    continue
                base_name = os.path.splitext(os.path.basename(wav_path))[0]
                npy_filename = f"{base_name}.npy"
                npy_path = os.path.join(output_dir, npy_filename)
                if os.path.exists(npy_path):
                    logging.warning(f"文件已存在，跳过保存: {npy_path}")
                    continue
                np.save(npy_path, embedding)
            else:
                logging.warning(f"No embedding found for {wav_path}")

        except Exception as e:
            logging.error(f"Error processing {wav_path}: {e}")

    logging.info("所有文件处理完成。")

if __name__ == "__main__":
    # 定义输入和输出目录
    input_directory = "dataset/ECD-TSE"
    output_directory = "dataset/embedding"

    extract_and_save_embeddings(input_directory, output_directory)
