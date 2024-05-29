import json,torch

def pack_data(json_file,engine_list):
    cases = json.load(open(json_file, 'r'))
    engines = engine_list
    data_dic = {'data': [
        {"question": each["question"],
         "golden_answer": each["golden_answer"],
         "answer": each[f"answer_{engine}"]} 
         for each in cases for engine in engines if not each['improper']],
                'target': [1 if each[f"judge_{engine}"] else 0 for each in cases for engine in engines if not each['improper']]}
    print(f"Total data points: {len(data_dic['data'])}")
    count_ones = sum(1 for i in data_dic['target'] if i == 1)
    print(f"Count of ones: {count_ones}")
    print(f"Count of zeros: {len(data_dic['target']) - count_ones}")
    torch.save(data_dic, json_file.replace('.json', '.pt'))


if __name__ =='__main__':
    #engine_list=['fid', 'gpt35', 'chatgpt', 'newbing', 'gpt4']
    # pack_data(json_file="./data/NQ.json")
    engine_list=['llava1.57b','llava1.513b','InstructBLIP7b','VisualGLM8b','MiniGPTV'] # 都是2000个，还有MiniGPT47b和openFlamingo29b BLIP7b只有1000个
    pack_data(json_file="./data/okvqa_test_2000.json",engine_list=engine_list)