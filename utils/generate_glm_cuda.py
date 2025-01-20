
import torch
import json
from tqdm import tqdm
import pandas as pd

def load_data():
    data = pd.read_excel("")    # This is a private dataset in our dataset. But it is not available for you due to the patient privacy issue. 
                                # Pleaese add the path of the public dataset here.
    result = data.loc[:, ['Index','case_description','diagnosis','differential_diagnosis_ch']]
    result = result.dropna(axis=0, how='any')
    return result

def generate_glm(tokenizer, model, device, patient_history, diagnosis, prompt_type="basic"):
    templates = {
        "basic": "患者的临床病例摘要将显示在三重分隔符中，您将使用此信息进行诊断（不是临床诊断），请描述多种鉴别诊断以及每种诊断的解释。以JSON格式输出结果，JSON的关键字是鉴别诊断的名称，值是每个鉴别诊断的解释。'''{}'''\请用中文输出结果：",
        "with_final_diag": "患者的临床病例摘要及所患疾病将显示在三重分隔符中，您将使用此信息进行诊断（不是临床诊断），请描述多种鉴别诊断以及每种诊断的解释。 以JSON格式输出结果，JSON的关键字是鉴别诊断的名称，值是每个鉴别诊断的解释。'''{}\n 患者所患疾病为：{}''' \n请用中文输出结果：",
        "with_format_ctrl": "患者的临床病例摘要将显示在三重分隔符中，您将使用此信息进行诊断（不是临床诊断），请描述多种鉴别诊断以及每种诊断的解释，鉴别诊断解释应包含支持点、不支持点、结论三部分。以JSON格式输出结果，JSON的关键字是鉴别诊断的名称，值是每个鉴别诊断的解释。'''{}''' \n请用中文输出结果：",
        "with_fd_fc": "患者的临床病例摘要及所患疾病将显示在三重分隔符中，您将使用此信息进行诊断（不是临床诊断），请描述多种鉴别诊断以及每种诊断的解释。鉴别诊断解释应包含支持点、不支持点、结论三部分。以JSON格式输出结果，JSON的关键字是鉴别诊断的名称，值是每个鉴别诊断的解释。'''{}\n 患者所患疾病为：{}''' \n请用中文输出结果：",
    }
    if prompt_type=="basic" or prompt_type=="with_format_ctrl":
        content = templates[prompt_type].format(patient_history)
    else:
        content = templates[prompt_type].format(patient_history, diagnosis)
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )
    inputs = inputs.to(device)

    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result_str = result.replace(" ",'').replace('json','').replace('\n','').replace("```",'')
    try:
        result_json = json.loads(result_str)
    except:
        print("Error occurs when changing string to json.")
        return {'default':result, 'string':result_str, 'json':'error'}
    return {'default':result, 'string':result_str, 'json':result_json}


def generate(tokenizer, model, data, prompt_type="basic", model_name="glm-4-9b"):
    ## load model
    device = torch.device('cuda')
    model.eval().to(device)

    ## process data
    idx = list(data['Index'])
    patient_histories = list(data['case_description'])
    diagnoses = list(data['diagnosis'])

    ## generation
    assert len(idx)==len(patient_histories)
    results = dict()
    for i in tqdm(range(len(idx))):
        results[idx[i]] = generate_glm(tokenizer, model, device, patient_histories[i], diagnoses[i], prompt_type)

    ## save the results
    json_str = json.dumps(results,ensure_ascii=False)
    with open("./output/"+prompt_type+"/"+"data_"+model_name+".json", 'w', encoding='utf-8') as f:
        f.write(json_str)