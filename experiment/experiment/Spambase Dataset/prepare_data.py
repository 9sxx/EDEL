import pandas as pd
import re

path_train = "../../dataset/Spambase Dataset/spambase.data"

# 原数据列名，由于列名中存在特殊字符，在使用xgboost建模时可能会遇到报错，故替换列名
columns = [
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
    'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail',
    'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses',
    'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
    'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp',
    'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs',
    'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
    'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
    'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
    'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(',
    'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average',
    'capital_run_length_longest', 'capital_run_length_total', 'class'
]

# 创建正则表达式对象来替换特殊字符
regex = re.compile(r"[\[\]<]", re.IGNORECASE)

# 清理列名
cleaned_columns = [regex.sub("_", col) for col in columns]


# 处理重复列名
def make_unique(column_names):
    seen = set()
    new_names = []
    for name in column_names:
        unique_name = name
        i = 1
        while unique_name in seen:
            unique_name = f"{name}_{i}"
            i += 1
        seen.add(unique_name)
        new_names.append(unique_name)
    return new_names


# 更新列名为唯一
columns = make_unique(cleaned_columns)

data = pd.read_csv(path_train, names=columns)
print(data)

# 获取数据集的形状
shape = data.shape
num_samples = shape[0]
num_features = shape[1]
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

# 查看不平衡程度
print(data['class'].value_counts())

# 导出处理后的特征和标签
data.to_csv("data.csv", index=False)

print("Data processing successful!")
