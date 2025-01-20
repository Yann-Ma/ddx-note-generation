from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import json

def load_data():
    data = pd.read_excel("")    # Add the dataset path here
    result = data.loc[:, ['Index','case_description','diagnosis','diagnosis_basis_ch','differential_diagnosis_ch','diagnosis_basis_en','differential_diagnosis_en']]
    result = result.dropna(axis=0, how='any')
    return result

def generate_gpt_ch(patient_history, diagnosis, prompt_type="basic", model_name="gpt-4o-mini"):
    client = OpenAI(api_key = '')   # Add your api key here
    templates = {
        "basic": "患者的临床病例摘要将显示在三重分隔符中，您将使用此信息进行诊断（不是临床诊断），请描述多种鉴别诊断以及每种诊断的解释。以JSON格式输出结果，JSON的关键字是鉴别诊断的名称，值是每个鉴别诊断的解释。'''{}'''\n输出：",
        "with_final_diag": "患者的临床病例摘要及所患疾病将显示在三重分隔符中，您将使用此信息进行诊断（不是临床诊断），请描述多种鉴别诊断以及每种诊断的解释。 以JSON格式输出结果，JSON的关键字是鉴别诊断的名称，值是每个鉴别诊断的解释。'''{}\n 患者所患疾病为：{}''' \n输出：",
        "with_format_ctrl": "患者的临床病例摘要将显示在三重分隔符中，您将使用此信息进行诊断（不是临床诊断），请描述多种鉴别诊断以及每种诊断的解释，鉴别诊断解释应包含支持点、不支持点、结论三部分。以JSON格式输出结果，JSON的关键字是鉴别诊断的名称，值是每个鉴别诊断的解释。'''{}''' \n输出：",
        "with_fd_fc": "患者的临床病例摘要及所患疾病将显示在三重分隔符中，您将使用此信息进行诊断（不是临床诊断），请描述多种鉴别诊断以及每种诊断的解释。鉴别诊断解释应包含支持点、不支持点、结论三部分。以JSON格式输出结果，JSON的关键字是鉴别诊断的名称，值是每个鉴别诊断的解释。'''{}\n 患者所患疾病为：{}''' \n输出：",
    }
    if prompt_type=="basic" or prompt_type=="with_format_ctrl":
        content = templates[prompt_type].format(patient_history)
    else:
        content = templates[prompt_type].format(patient_history, diagnosis)

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是一名经验丰富的医生"},
            {
                "role": "user",
                "content": content
            }
        ]
    )

    result = completion.choices[0].message.content
    result_str = result.replace('\n','').replace(' ', '').replace('，', ',').replace('json','').replace("```", '')
    try:
        result_json = json.loads(result_str)
    except:
        print("Error occurs when changing string to json.")
        return {'default':result, 'string':result_str, 'json':'error'}
    return {'default':result, 'string':result_str, 'json':result_json}

def generate(data, prompt_type="basic", model_name="gpt-4o-mini"):
    ## process data
    idx = list(data['Index'])
    patient_histories = list(data['case_description'])
    diagnoses = list(data['diagnosis'])

    ## generation
    assert len(idx)==len(patient_histories)
    results = dict()
    for i in tqdm(range(len(idx))):
        results[idx[i]] = generate_gpt_ch(patient_histories[i], diagnoses[i], prompt_type, model_name)

    ## save the results
    json_str = json.dumps(results,ensure_ascii=False)
    with open("/Users/yiyi/workspace/output/"+prompt_type+"/"+"data_"+model_name+".json", 'w', encoding='utf-8') as f:
        f.write(json_str)

