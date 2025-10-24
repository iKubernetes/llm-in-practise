import json

# 定义你想要替换的名称和作者
REPLACEMENTS = {
    'zh': {  # 中文替换值
        '{{NAME}}': '马哥教育AI小助手',  # 例如：马哥教育智能助理
        '{{AUTHOR}}': '马哥教育AI团队'  # 例如：马哥教育AI团队
    },
    'en': {  # 英文替换值（如果数据集中有英文部分）
        '{{NAME}}': 'MageEdu AI',  # 例如：Little_Q
        '{{AUTHOR}}': 'MageEdu AI Team'  # 例如：QFans
    }
}

def convert_to_alpaca_format(data):
    # 根据语言标签获取对应的替换字典
    replacements = REPLACEMENTS[data['tag']]

    # 处理响应，替换占位符
    response = data['response']
    for placeholder, value in replacements.items():
        response = response.replace(placeholder, value)

    # 构建并返回Alpaca格式的数据
    return {
        'instruction': data['query'],  # 原始数据中的'query'对应Alpaca的'instruction'
        'input': '',       # 该数据集没有额外的input字段，留空
        'output': response, # 替换后的回答作为'output'
        'system': '',      # 该数据集没有系统提示词，留空
        'history': []      # 该数据集是单轮问答，没有历史对话
    }

def main():
    converted_data = []
    # 读取原始的JSONL文件
    with open('self_cognition.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                original_data = json.loads(line)
                # 转换每一行数据
                converted_data.append(convert_to_alpaca_format(original_data))

    # 将转换后的数据保存为JSON文件
    with open('self_cognition_alpaca.json', 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
