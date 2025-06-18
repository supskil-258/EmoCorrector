import os
import re
from data_gen.tts.base_preprocess import BasePreprocessor


class AzurePreAlign(BasePreprocessor):

    def meta_data(self):
        emotions = ['angry', 'fear', 'sadness', 'happy', 'neutral']
        pattern = re.compile('[\t\n ]+')  # 用于清理多余的空白符

        # 获取最外层的文件夹（001到010）
        spk_folders = [f for f in os.listdir(self.raw_data_dir) if
                       os.path.isdir(os.path.join(self.raw_data_dir, f)) and re.match(r'^\d{3}$', f)]

        item_names = set()  # 用于检查唯一性

        # 遍历每个说话人文件夹
        for spk_name in spk_folders:
            spk_folder = os.path.join(self.raw_data_dir, spk_name)

            # 遍历每个情感文件夹
            for emotion in emotions:
                emotion_folder = os.path.join(spk_folder, emotion)
                text_folder = os.path.join(emotion_folder, 'text')
                audio_folder = os.path.join(emotion_folder, 'raw')

                # 遍历文本文件
                for text_file in os.listdir(text_folder):
                    if text_file.endswith('.txt'):
                        item_name = os.path.splitext(text_file)[0]

                        # 检查 item_name 是否唯一
                        if item_name in item_names:
                            print(
                                f"Warning: Duplicate item_name found: {item_name} in {spk_name} for emotion {emotion}")
                        item_names.add(item_name)

                        # 读取文本文件内容
                        with open(os.path.join(text_folder, text_file), 'r') as f:
                            line = f.readline().strip()
                            line = re.sub(pattern, ' ', line)  # 清理多余空格

                            if line:  # 检查非空行
                                clean_wav_fn = os.path.join(audio_folder, f'{item_name}.wav')


                                # 生成包含说话人和情感信息的数据
                                yield {
                                    'item_name': item_name,  # 使用原始 item_name
                                    'spk_name': spk_name,  # 添加说话人名称
                                    'clean_wav_fn': clean_wav_fn,
                                    'txt': line,
                                    'emotion': emotion,

                                }


if __name__ == "__main__":
    AzurePreAlign().process()
