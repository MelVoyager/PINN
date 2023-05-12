import argparse
import os, sys
import numpy as np
import glob
import Levenshtein
from Burgers_1D.test import burgers_1d
from Heat_2d_multi_scale.test import heat_2d_multi_scale
from Poisson2D.test import poisson2d
from Poisson3D.test import poisson3d
from Poisson_Boltzmann2D.test import poisson_boltzmann2d
from Wave.test import wave
from Heat_Darcy.test import heat_darcy
os.chdir(sys.path[0])

def get_latest_file(folder_path):
    # 获取文件夹中所有文件的列表
    files = glob.glob(os.path.join(folder_path, "*"))

    # 按修改时间从新到旧排序
    files.sort(key=os.path.getmtime, reverse=True)

    # 返回最新文件的地址（如果文件夹非空）
    return files[0] if files else None

def read_single_png_file(folder_path):
    # 获取文件夹中的第一个以.png结尾的文件
    png_file = glob.glob(os.path.join(folder_path, "*.png"))[0]

    # 以二进制方式读取文件
    with open(png_file, "rb") as f:
        content = f.read()

    return os.path.basename(png_file), content

def save_single_png_file(file, output_folder):
    file_name, content = file
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, file_name)
    with open(output_path, "wb") as f:
        f.write(content)
        
directorys = ['Burgers_1D', 
              'Heat_Darcy', 'Heat_2d_multi_scale', 
              'Poisson2D', 'Poisson3D', 'Poisson_Boltzmann2D',
              'Wave']

functions = [burgers_1d,
             heat_darcy, heat_2d_multi_scale,
             poisson2d, poisson3d, poisson_boltzmann2d,
             wave]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is a program to run specific model of different equations.')

    namelist = "', '".join(directorys)
    namelist = "'" + namelist + "'"
    parser.add_argument('-name', type=str, help='The input is expected to a subset of ' + namelist)
    args = parser.parse_args()
    if args.name:
        names = args.name.split(',')
        names = [item.strip() for item in names]
        # print(names)
        indices = []
        RESET = "\033[0m"
        RED = "\033[31m"
        GREEN = "\033[32m"
        for i in range(len(names)):
            try:
                indices.append(directorys.index(names[i]))
            except ValueError as e:
                print('Your input contain model name which cannot be recognized.')
                print(f'Your input:    {RED} {names[i]} {RESET}')
                print(f'Expected input:{GREEN} {min(directorys, key=lambda word: Levenshtein.distance(word, names[i]))} {RESET}')
                exit(0)
    else:
        indices = range(0, len(directorys))
    
    for i in indices:
        err = [0, 0, 0]
        models = [None, None, None]
        figs = [None, None, None]
        os.chdir('./'+directorys[i])
        for j in range(3):
            print(f'{directorys[i]}_{j+1}')
            # print(os.getcwd())
            ret = functions[i]()
            err[j] = ret.item()
            with open(get_latest_file('./model'), "rb") as source_file:
                models[j] = source_file.read()
            figs[j] = read_single_png_file('.')
        err = np.array(err)
        min_index = np.argmin(err)
        with open(get_latest_file('./model'), "wb") as destination_file:
            destination_file.write(models[min_index])
        save_single_png_file(figs[min_index], '.')
        os.chdir('./..')
    
