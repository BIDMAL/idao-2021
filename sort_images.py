from tqdm import tqdm
import os
import shutil

dir_name = './idao/idao_dataset/'
for type_dir in ['ER', 'NR']:
    source_dir = os.path.join(dir_name + type_dir)
    for class_name in [1, 3, 6, 10, 20, 30]:
        os.makedirs(os.path.join(source_dir, str(class_name)), exist_ok=True)

for type_dir in ['ER', 'NR']:
    source_dir = os.path.join(dir_name + '/train/', type_dir)
    for i, file_name in enumerate(tqdm(os.listdir(source_dir))):
        if file_name[0] == '.':
            continue
        if type_dir == 'ER':
            i = 6
        else:
            i = 7
        energy = file_name.split('_')[i]

        data_save_path = dir_name + type_dir + '/' + energy + '/' + file_name
        shutil.copy(os.path.join(source_dir, file_name), data_save_path)
