import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='这是一个示例程序')

# 添加参数
parser.add_argument('-name', type=str, help='name参数的帮助信息')

# 解析参数
args = parser.parse_args()

# 使用参数
names = args.name.split(', ')
print(names)
