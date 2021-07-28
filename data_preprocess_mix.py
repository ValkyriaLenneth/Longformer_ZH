"""
Preprocess the Chinese Pretrain Datasets:
    1. wikizh: json: "text"                         1,043,224
    2. news2016zh: "title" + "content" + "desc"     2,500,000
    3. baike2018qa: "title" + "desc" + "answer"     1,500,000
    4. webtext2019zh: "title" + "desc" + "content"  4,100,000
return:
    the text file which contains all entries in the dataset.

data:
    each line is a JSON object

flow:
    1. Change the format of all files in the dataset into ".json"
    2. deal with each json file and extract the tag 'text' for each entry with /t
    3. return the output data
    4. split the training data and valid data
"""
import os
import json
import random
import jionlp
import argparse
import math
from tqdm import tqdm, trange

def main(dataset):
    wiki_path = r'dataset/wiki_zh_2019'
    baike_path = r'dataset/baike2018qa'
    webtext_path = r'dataset/webtext2019zh'
    news_path = r'dataset/new2016zh'

    if dataset == 'wiki':
        file_path = r'dataset/wiki_zh_2019'
    elif dataset == 'baike':
        file_path = r'dataset/baike2018qa'
    elif dataset == 'news':
        file_path = r'dataset/new2016zh'
    elif dataset == 'webtext':
        file_path = r'dataset/webtext2019zh'
    elif dataset == 'mix':
        file_path = [wiki_path, webtext_path, news_path]

    output_path = r'./dataset/preprocessed/'
    
    
    total_texts = []
    max_number_per_set = 300000
    for t, path in enumerate(file_path):
        texts = []
        print('Preprocess: ', path)
        for dir_path, dir_name, files in os.walk(path):
            for file in files:
                print('Current: ', file)
                with open(dir_path + '/' + file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for i, line in enumerate(tqdm(lines)):
                        if len(texts) >= max_number_per_set:
                            break

                        j = json.loads(line)
                        if file.startswith('wiki'): # N
                            text = j['text']
                        elif file.startswith('news'): # 2.5M
                            text = j['content']
                        elif file.startswith('baike'): 
                            text = j['title'] + j['answer']
                        elif file.startswith('web'): # 4.3M
                            text = j['title'] + j['content']
                        else:
                            continue
                        
                        text = jionlp.clean_text(text)
                        texts.append(text)
        total_texts.extend(texts)
                            
    

    random.shuffle(total_texts)

    print('total samples: ', len(total_texts))
    print('Writing into file: ', output_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    
    val_set_output = output_path + dataset + '_val'
    val_num = 3000
    print('generating valid set')
    with open(val_set_output, 'w', encoding='utf-8') as w:
        w.writelines(total_texts[0: val_num])

    print('writing train set')
    train_set_output = output_path + dataset + '_train'
    with open(train_set_output, 'w', encoding='utf-8') as w:
        w.writelines(total_texts[val_num:])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='mix', required=True, help='choose the dataset')
    args = parser.parse_args()

    main(args.dataset)

