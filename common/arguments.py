import argparse


def prase_args():
    parser = argparse.ArgumentParser(description='Args') # 创建解析器

    parser.add_argument('-config', '--config_file', default='./config_file/config_okvqa.yaml', type=str, help='the location of the config file path')
    args = parser.parse_args()
    return args