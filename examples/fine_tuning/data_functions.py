from datasets import load_dataset, concatenate_datasets
from helper_functions import *
import torch
from trl.trainer import ConstantLengthDataset
from omegaconf import OmegaConf
import random
import glob
import json
from torch.utils.data import IterableDataset
from datasets import disable_caching

disable_caching()


def split_dataset(args, dataset):
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
    else:
        dataset = dataset.train_test_split(test_size=0.000001, seed=None)
        train_data = dataset["train"]
        valid_data = dataset["test"]
    return train_data, valid_data


prompt_format_without_context = (
    """### USER: {instruction}\n\n### RESPONSE: {response}"""
)

# prompt_format_without_context = """### USER:
# {instruction}

# ### RESPONSE:
# {response}"""

# prompt_format_with_context = """### CONTEXT:
# {context}

# ### USER:
# {instruction}

# ### RESPONSE:
# {response}"""


# def add_col(a):
#   a["input"] = ""
#   return a

# def add_col12(a):
#   a["system_prompt"] = ""
#   return a

# def add_colume_more_data(example):

#     example["bot_morethan_one"] = 2
#     example["has_context"] = 1 if example["input"] != "" else 0

#     comb  = example["system_prompt"] + " "+ example["instruction"]
#     example["Context"] = example["input"]
#     example["Instruction"] = comb.strip()
#     example["Answer"] =  example["output"]

#     return example


def deterministic_random(seed: int) -> random.Random:
    return random.Random(seed)


# def add_more_dataset():

#     dataset1 = load_dataset("garage-bAInd/Open-Platypus")
#     # dataset2 = load_dataset("ehartford/dolphin",data_files="flan1m-alpaca-uncensored.jsonl")
#     # dataset3 = load_dataset("Open-Orca/OpenOrca")

#     dataset1 = dataset1.remove_columns("data_source")


#     # random_test_ids = deterministic_random(42).sample(range(len(dataset2["train"])), k=50000)
#     # dataset2_1persent = dataset2["train"].select(random_test_ids)

#     # random_test_ids = deterministic_random(42).sample(range(dataset3["train"]), k=50000)
#     # dataset3_1persent = dataset3["train"].select(random_test_ids).rename_column("question", "instruction").rename_column("response", "output")

#     # dataset3_1persent = dataset3_1persent.remove_columns("id")
#     # dataset3_1persent = dataset3_1persent.map(add_col)

#     dataset1_fullpercent = dataset1.map(add_col12)
#     #dataset2_1persent = dataset2_1persent.map(add_col12)

#     #all_dataset = concatenate_datasets([dataset1_fullpercent, dataset2_1persent, dataset3_1persent])

#     all_dataset = dataset1_fullpercent.map(add_colume_more_data)

#     return all_dataset.remove_columns(['input', 'output', 'instruction', 'system_prompt', 'bot_morethan_one'])


def remove_not_relate_to_thaiEn_dolphin(example):
    a = example["input"]
    fullstring = example["input"].lower()
    substring1 = "language"  # ["language" , "translate" , "english"]
    substring2 = "translate"
    # substring3 = "english"

    if substring1 in fullstring or substring2 in fullstring:
        example["has_tran"] = 1
    else:
        example["has_tran"] = 0

    ins = example["instruction"] + " " + example["input"]
    example["Answer"] = example["output"].strip()
    example["Context"] = ""
    example["Instruction"] = ins.strip()

    return example


def re_formate_iappQA(example):
    # combine = "เนื้อหาบทความ (context): " + example["context"].strip() +"\nคำถาม (question): จากเนื้อหาบทความ " + example["question"].strip()
    combine = (
        "พื้นหลัง: "
        + example["context"].strip()
        + "\nคำถาม: "
        + example["question"].strip()
    )

    example["Answer"] = "ตอบ: " + example["answers"]["text"][0].strip()
    example["Context"] = ""
    example["Instruction"] = combine.strip()

    return example


def re_formate_thaisum(example):
    #    ch_tags = ""
    #    if example["tags"].strip() != "" :
    #       ch_tags = " คำสำคัญ หรือ Tags ที่ควรจะเป็นคือ "+example["tags"].strip()

    #    if ch_tags != "":
    #         combine = "เนื้อหาข่าว (news): " + example["body"].strip()+ "\nจากเนื้อหาข่าว จงสรุปเนื้อหาให้กระชับ เข้าใจง่าย พร้อมทั้งบอกคำสำคัญ หรือ Tags"
    #    else:
    #         combine = "เนื้อหาข่าว (news): " + example["body"].strip()+ "\nจากเนื้อหาข่าว จงสรุปเนื้อหาให้กระชับ เข้าใจง่าย"

    combine = (
        "เนื้อหาข่าว (news): "
        + example["body"].strip()
        + "\nจากเนื้อหาข่าว จงสรุปเนื้อหาให้กระชับ เข้าใจง่าย"
    )

    example["Answer"] = example["summary"].strip()  # + ch_tags
    example["Context"] = ""
    example["Instruction"] = combine.strip()

    return example


def re_formate_xlsum(example):
    #    ch_title = ""
    #    if example["title"].strip() != "" :
    #       ch_title = "\nหัวข้อหรือ (title) ที่เหมาะสมคือ "+example["title"].strip()

    #    if ch_title != "":
    #         combine = "เนื้อหา บทความ (article): " + example["text"].strip()+ "\nจากเนื้อหาบทความที่กล่าวก่อนหน้านี้ จงสรุปบทความโดยมีเนื้อหาที่สั้นและเข้าใจง่าย พร้อมทั้งตั้งชื่อ หัวข้อหรือ (title) ที่เหมาะสมกับบทความ"
    #    else:
    #         combine = "เนื้อหา บทความ (article): " + example["text"].strip()+ "\nจากเนื้อหาบทความที่กล่าวก่อนหน้านี้ จงสรุปบทความโดยมีเนื้อหาที่สั้นและเข้าใจง่าย"

    combine = (
        "เนื้อหา บทความ (article): "
        + example["text"].strip()
        + "\nจากเนื้อหาบทความที่กล่าวก่อนหน้านี้ จงสรุปบทความโดยมีเนื้อหาที่สั้นและเข้าใจง่าย"
    )

    example["Answer"] = example["summary"].strip()  # + ch_title
    example["Context"] = ""
    example["Instruction"] = combine.strip()

    return example


def re_formate_scb_then(example):
    example["Answer"] = example["translation"]["en"].strip()
    example["Context"] = ""
    example["Instruction"] = (
        "Translate Thai to English. จงแปลภาษาไทยเป็นอังกฤษ\n"
        + example["translation"]["th"].strip()
    )

    return example


def re_formate_scb_enth(example):
    example["Answer"] = example["translation"]["th"]
    example["Context"] = ""
    example["Instruction"] = (
        "Translate English to Thai. จงแปลภาษาอังกฤษเป็นภาษาไทย\n"
        + example["translation"]["en"].strip()
    )

    return example


def re_formate_xp3(example):
    example["Answer"] = example["targets"].strip()
    example["Context"] = ""
    example["Instruction"] = example["inputs"].strip()

    return example


def re_formate_cot(example):
    example["Answer"] = example["rationale"].strip()
    example["Context"] = ""
    example["Instruction"] = example["source"].strip()
    return example


def read_file():
    path = "/workspace/sealion/examples/xp3x/content/drive/MyDrive/xp3/*"
    all_file = []
    j = 0
    for file in glob.glob(path):
        j = j + 1
        try:
            with open(file) as f:
                data = [json.loads(line) for line in f]

                all_file.append(file)
        except Exception as e:
            # print(e)
            continue
    print("all file is ", j)
    return all_file


def re_formate_dolly(example):
    if example["context"] != "":
        comb = example["instruction"] + "\n" + example["context"]
    else:
        comb = example["instruction"]

    example["Answer"] = example["response"].strip()
    example["Context"] = ""
    example["Instruction"] = comb.strip()

    return example


def re_formate_han(example):
    example["Answer"] = "ตอบ: " + example["a"].strip()
    example["Context"] = ""
    example["Instruction"] = "คำถาม: " + example["q"].strip()

    return example


def re_formate_code(example):
    example["Answer"] = example["response"]
    example["Context"] = ""
    example["Instruction"] = example["prompt"].strip()
    return example


def re_formate_platypus(example):
    if example["input"] != "":
        comb = example["instruction"] + "\n" + example["input"]
    else:
        comb = example["instruction"]

    example["Answer"] = example["output"]
    example["Context"] = ""
    example["Instruction"] = comb.strip()
    return example


def re_formate_indoqa(example):
    combine = (
        "Context (เนื้อหา): "
        + example["context"].strip()
        + "\nQuestion (คำถาม): "
        + example["question"].strip()
    )

    example["Answer"] = example["answer"].strip()
    example["Context"] = ""
    example["Instruction"] = combine
    return example


def re_formate_COIG_china(example):
    if example["instruction"] != "":
        combine = (
            example["instruction"].strip()
            + "\n問題: "
            + example["conversations"][0]["question"].strip()
        )
    else:
        combine = "問題: " + example["conversations"][0]["question"].strip()

    example["Answer"] = "答案: " + example["conversations"][0]["answer"].strip()
    example["Context"] = ""
    example["Instruction"] = combine
    return example


def re_formate_thai_sample_500k(example):
    example["Answer"] = ""
    example["Context"] = ""
    example["Instruction"] = example["text"].strip()
    return example


def load_dolly_V15(args, split_train_test=True):  # V13,V14,V15
    dataset = load_dataset(
        "pythainlp/final_training_set_v1",
        split="train",
        num_proc=8 if not args.streaming else None,
        streaming=args.streaming,
    )

    dataset = dataset.map(add_prefix)
    dataset = dataset.filter(lambda x: x["bot_morethan_one"] == 2)
    dataset = dataset.remove_columns(
        ["has_context", "bot_morethan_one", "text", "metadata", "nb_token"]
    )

    dataset_dolphin = load_dataset(
        "ehartford/dolphin", data_files="flan1m-alpaca-uncensored.jsonl", split="train"
    )
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_dolphin)), k=10000
    )
    dataset_dolphin = dataset_dolphin.select(random_test_ids)
    dataset_dolphin = dataset_dolphin.map(remove_not_relate_to_thaiEn_dolphin)
    dataset_dolphin = dataset_dolphin.filter(lambda x: x["has_tran"] == 0)
    dataset_dolphin = dataset_dolphin.remove_columns(
        ["instruction", "input", "output", "has_tran"]
    )

    dataset_iapp = load_dataset("iapp_wiki_qa_squad", split="train")
    dataset_iapp = dataset_iapp.map(re_formate_iappQA)
    dataset_iapp = dataset_iapp.remove_columns(
        ["question_id", "article_id", "title", "context", "question", "answers"]
    )

    dataset_thaisum = load_dataset("thaisum", split="train")
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_thaisum)), k=5000
    )
    dataset_thaisum = dataset_thaisum.select(random_test_ids)
    dataset_thaisum = dataset_thaisum.map(re_formate_thaisum)
    dataset_thaisum = dataset_thaisum.remove_columns(
        ["title", "body", "summary", "type", "tags", "url"]
    )

    dataset_xlsum = load_dataset("csebuetnlp/xlsum", "thai", split="train")
    random_test_ids = deterministic_random(42).sample(range(len(dataset_xlsum)), k=5000)
    dataset_xlsum = dataset_xlsum.select(random_test_ids)
    dataset_xlsum = dataset_xlsum.map(re_formate_xlsum)
    dataset_xlsum = dataset_xlsum.remove_columns(
        ["id", "url", "title", "summary", "text"]
    )

    dataset_enth = load_dataset("scb_mt_enth_2020", "enth", split="train")

    random_test_ids = deterministic_random(42).sample(range(len(dataset_enth)), k=10000)
    dataset_enth = dataset_enth.select(random_test_ids)
    dataset_then = dataset_enth

    dataset_enth = dataset_enth.map(re_formate_scb_enth)
    dataset_enth = dataset_enth.remove_columns(["translation", "subdataset"])

    # dataset_then = load_dataset("scb_mt_enth_2020" , "then",split="train")
    # dataset_then = dataset_then.select(random_test_ids)
    dataset_then = dataset_then.map(re_formate_scb_then)
    dataset_then = dataset_then.remove_columns(["translation", "subdataset"])

    # dataset_alpaca_gpt4 = load_dataset("json", data_files="/workspace/sealion/alpaca_gpt4_data.json")
    # dataset_alpaca_gpt4 = dataset_alpaca_gpt4.rename_column("instruction", "Instruction").rename_column("output", "Answer").rename_column("input", "Context")
    # #data_add = add_more_dataset()
    # # print(data_add)
    # print(dataset)
    # print(dataset_dolphin)
    # print(dataset_iapp)
    # print(dataset_thaisum)
    # print(dataset_xlsum)
    # print(dataset_enth)
    # print(dataset_then)
    # print(dataset_alpaca_gpt4)

    dataset_han = load_dataset(
        "csv",
        data_files="/workspace/sealion/examples/pythainlp-han_dataset - pythainlp-han_dataset.csv",
    )
    dataset_han = dataset_han.map(re_formate_han).remove_columns(["q", "a"])

    all_file = read_file()
    dataset_ = load_dataset("json", data_files=all_file, split="train")
    dataset_1 = dataset_.filter(lambda example: example["split"] == "train")

    dataset_en_th = dataset_1.filter(
        lambda example: example["config"] in ["en_th"]
    )  # ['en_th', 'th', 'thai' ]
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_en_th)), k=40000
    )
    dataset_en_th = dataset_en_th.select(random_test_ids)
    dataset_en_th = dataset_en_th.map(re_formate_xp3)
    dataset_en_th = dataset_en_th.remove_columns(
        ["inputs", "targets", "language", "split", "template", "dataset", "config"]
    )

    dataset_thai = dataset_1.filter(lambda example: example["config"] in ["thai"])
    # random_test_ids = deterministic_random(42).sample(range(len(dataset_thai)), k=80000)
    # dataset_thai = dataset_thai.select(random_test_ids)
    dataset_thai = dataset_thai.map(re_formate_xp3)
    dataset_thai = dataset_thai.remove_columns(
        ["inputs", "targets", "language", "split", "template", "dataset", "config"]
    )

    dataset_Platypus = load_dataset("garage-bAInd/Open-Platypus", split="train")
    dataset_Platypus = dataset_Platypus.filter(
        lambda example: example["data_source"] != "scienceqa"
        and example["data_source"] != "reclor"
    )  # due to copyright
    dataset_Platypus = dataset_Platypus.map(re_formate_platypus)
    dataset_Platypus = dataset_Platypus.remove_columns(
        ["input", "output", "instruction", "data_source"]
    )

    # dataset_dolly = load_dataset(
    #         "databricks/databricks-dolly-15k",
    #         split="train",
    #         )
    # dataset_dolly = dataset_dolly.map(re_formate_dolly)
    # dataset_dolly = dataset_dolly.remove_columns(['instruction', 'context', 'response', 'category'])

    # dataset_indoqa = load_dataset("jakartaresearch/indoqa", split="train")
    # dataset_indoqa = dataset_indoqa.filter(lambda example: example["category"] == "SPAN")
    # dataset_indoqa = dataset_indoqa.map(re_formate_indoqa)
    # dataset_indoqa = dataset_indoqa.remove_columns(['id', 'context', 'question', 'answer', 'category', 'span_start', 'span_end'])

    dataset_COIG_china = load_dataset("BAAI/COIG", split="NoTranslate")
    dataset_COIG_china = dataset_COIG_china.map(re_formate_COIG_china)
    dataset_COIG_china = dataset_COIG_china.remove_columns(
        ["instruction", "conversations"]
    )

    print(dataset)
    print(dataset_dolphin)
    print(dataset_iapp)
    print(dataset_thaisum)
    print(dataset_xlsum)
    print(dataset_enth)
    print(dataset_then)
    print(dataset_en_th)
    print(dataset_enth)
    print(dataset_thai)
    print(dataset_han["train"])
    print(dataset_Platypus)
    print(dataset_COIG_china)

    dataset = concatenate_datasets(
        [
            dataset,
            dataset_dolphin,
            dataset_iapp,
            dataset_thaisum,
            dataset_xlsum,
            dataset_enth,
            dataset_then
            # , dataset_alpaca_gpt4["train"],
            ,
            dataset_en_th,
            dataset_thai
            # dataset_dolly,
            ,
            dataset_han["train"],
            dataset_Platypus,
            dataset_COIG_china,
        ]
    )

    dataset = dataset.map(prepare_dolly_text)
    # print(dataset)

    if split_train_test:
        return split_dataset(args, dataset)
    else:
        return dataset


def load_dolly_V12(args, split_train_test=True):  # V12
    dataset_iapp = load_dataset("iapp_wiki_qa_squad", split="train")
    dataset_iapp = dataset_iapp.map(re_formate_iappQA)
    dataset_iapp = dataset_iapp.remove_columns(
        ["question_id", "article_id", "title", "context", "question", "answers"]
    )

    dataset_han = load_dataset(
        "csv",
        data_files="/workspace/sealion/examples/pythainlp-han_dataset - pythainlp-han_dataset.csv",
    )
    dataset_han = dataset_han.map(re_formate_han).remove_columns(["q", "a"])

    dataset = concatenate_datasets([dataset_iapp, dataset_han["train"]])

    # dataset = dataset.map(prepare_dolly_text)

    if split_train_test:
        return split_dataset(args, dataset)
    else:
        return dataset


def load_dolly_V16(args, split_train_test=True):  # V16
    dataset = load_dataset(
        "pythainlp/final_training_set_v1",
        split="train",
        num_proc=8 if not args.streaming else None,
        streaming=args.streaming,
    )

    dataset = dataset.map(add_prefix)
    dataset = dataset.filter(lambda x: x["bot_morethan_one"] == 2)
    dataset = dataset.remove_columns(
        ["has_context", "bot_morethan_one", "text", "metadata", "nb_token"]
    )

    dataset_dolphin = load_dataset(
        "ehartford/dolphin", data_files="flan1m-alpaca-uncensored.jsonl", split="train"
    )
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_dolphin)), k=10000
    )
    dataset_dolphin = dataset_dolphin.select(random_test_ids)
    dataset_dolphin = dataset_dolphin.map(remove_not_relate_to_thaiEn_dolphin)
    dataset_dolphin = dataset_dolphin.filter(lambda x: x["has_tran"] == 0)
    dataset_dolphin = dataset_dolphin.remove_columns(
        ["instruction", "input", "output", "has_tran"]
    )

    dataset_iapp = load_dataset("iapp_wiki_qa_squad", split="train")
    dataset_iapp = dataset_iapp.map(re_formate_iappQA)
    dataset_iapp = dataset_iapp.remove_columns(
        ["question_id", "article_id", "title", "context", "question", "answers"]
    )

    dataset_thaisum = load_dataset("thaisum", split="train")
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_thaisum)), k=5000
    )
    dataset_thaisum = dataset_thaisum.select(random_test_ids)
    dataset_thaisum = dataset_thaisum.map(re_formate_thaisum)
    dataset_thaisum = dataset_thaisum.remove_columns(
        ["title", "body", "summary", "type", "tags", "url"]
    )

    dataset_xlsum = load_dataset("csebuetnlp/xlsum", "thai", split="train")
    random_test_ids = deterministic_random(42).sample(range(len(dataset_xlsum)), k=5000)
    dataset_xlsum = dataset_xlsum.select(random_test_ids)
    dataset_xlsum = dataset_xlsum.map(re_formate_xlsum)
    dataset_xlsum = dataset_xlsum.remove_columns(
        ["id", "url", "title", "summary", "text"]
    )

    dataset_enth = load_dataset("scb_mt_enth_2020", "enth", split="train")

    random_test_ids = deterministic_random(42).sample(range(len(dataset_enth)), k=10000)
    dataset_enth = dataset_enth.select(random_test_ids)
    dataset_then = dataset_enth

    dataset_enth = dataset_enth.map(re_formate_scb_enth)
    dataset_enth = dataset_enth.remove_columns(["translation", "subdataset"])

    # dataset_then = load_dataset("scb_mt_enth_2020" , "then",split="train")
    # dataset_then = dataset_then.select(random_test_ids)
    dataset_then = dataset_then.map(re_formate_scb_then)
    dataset_then = dataset_then.remove_columns(["translation", "subdataset"])

    # dataset_alpaca_gpt4 = load_dataset("json", data_files="/workspace/sealion/alpaca_gpt4_data.json")
    # dataset_alpaca_gpt4 = dataset_alpaca_gpt4.rename_column("instruction", "Instruction").rename_column("output", "Answer").rename_column("input", "Context")
    # #data_add = add_more_dataset()
    # # print(data_add)
    # print(dataset)
    # print(dataset_dolphin)
    # print(dataset_iapp)
    # print(dataset_thaisum)
    # print(dataset_xlsum)
    # print(dataset_enth)
    # print(dataset_then)
    # print(dataset_alpaca_gpt4)

    dataset_han = load_dataset(
        "csv",
        data_files="/workspace/sealion/examples/pythainlp-han_dataset - pythainlp-han_dataset.csv",
    )
    dataset_han = dataset_han.map(re_formate_han).remove_columns(["q", "a"])

    all_file = read_file()
    dataset_ = load_dataset("json", data_files=all_file, split="train")
    dataset_1 = dataset_.filter(lambda example: example["split"] == "train")

    dataset_en_th = dataset_1.filter(
        lambda example: example["config"] in ["en_th"]
    )  # ['en_th', 'th', 'thai' ]
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_en_th)), k=40000
    )
    dataset_en_th = dataset_en_th.select(random_test_ids)
    dataset_en_th = dataset_en_th.map(re_formate_xp3)
    dataset_en_th = dataset_en_th.remove_columns(
        ["inputs", "targets", "language", "split", "template", "dataset", "config"]
    )

    dataset_thai = dataset_1.filter(lambda example: example["config"] in ["thai"])
    # random_test_ids = deterministic_random(42).sample(range(len(dataset_thai)), k=80000)
    # dataset_thai = dataset_thai.select(random_test_ids)
    dataset_thai = dataset_thai.map(re_formate_xp3)
    dataset_thai = dataset_thai.remove_columns(
        ["inputs", "targets", "language", "split", "template", "dataset", "config"]
    )

    dataset_Platypus = load_dataset("garage-bAInd/Open-Platypus", split="train")
    dataset_Platypus = dataset_Platypus.filter(
        lambda example: example["data_source"] != "scienceqa"
        and example["data_source"] != "reclor"
    )  # due to copyright
    dataset_Platypus = dataset_Platypus.map(re_formate_platypus)
    dataset_Platypus = dataset_Platypus.remove_columns(
        ["input", "output", "instruction", "data_source"]
    )

    # dataset_dolly = load_dataset(
    #         "databricks/databricks-dolly-15k",
    #         split="train",
    #         )
    # dataset_dolly = dataset_dolly.map(re_formate_dolly)
    # dataset_dolly = dataset_dolly.remove_columns(['instruction', 'context', 'response', 'category'])

    dataset_thai_sample_500k = load_dataset(
        "wannaphong/thai_sample_500k", split="train"
    )
    dataset_thai_sample_500k = dataset_thai_sample_500k.map(re_formate_thai_sample_500k)
    dataset_thai_sample_500k = dataset_thai_sample_500k.remove_columns(["text"])

    print(dataset)
    print(dataset_dolphin)
    print(dataset_iapp)
    print(dataset_thaisum)
    print(dataset_xlsum)
    print(dataset_enth)
    print(dataset_then)
    print(dataset_en_th)
    print(dataset_enth)
    print(dataset_thai)
    print(dataset_han["train"])
    print(dataset_Platypus)
    print(dataset_thai_sample_500k)

    dataset = concatenate_datasets(
        [
            dataset,
            dataset_dolphin,
            dataset_iapp,
            dataset_thaisum,
            dataset_xlsum,
            dataset_enth,
            dataset_then
            # , dataset_alpaca_gpt4["train"],
            ,
            dataset_en_th,
            dataset_thai
            # dataset_dolly,
            ,
            dataset_han["train"],
            dataset_Platypus,
            dataset_thai_sample_500k,
        ]
    )

    dataset = dataset.map(prepare_dolly_text)
    # print(dataset)

    if split_train_test:
        return split_dataset(args, dataset)
    else:
        return dataset


def load_dolly(args, split_train_test=True):  # V6
    dataset = load_dataset(
        "pythainlp/final_training_set_v1",
        split="train",
        num_proc=8 if not args.streaming else None,
        streaming=args.streaming,
    )

    dataset = dataset.map(add_prefix, load_from_cache_file=False)
    dataset = dataset.filter(lambda x: x["bot_morethan_one"] == 2)
    dataset = dataset.remove_columns(
        ["has_context", "bot_morethan_one", "text", "metadata", "nb_token"]
    )

    dataset_dolphin = load_dataset(
        "ehartford/dolphin", data_files="flan1m-alpaca-uncensored.jsonl", split="train"
    )
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_dolphin)), k=10000
    )
    dataset_dolphin = dataset_dolphin.select(random_test_ids)
    dataset_dolphin = dataset_dolphin.map(
        remove_not_relate_to_thaiEn_dolphin, load_from_cache_file=False
    )
    dataset_dolphin = dataset_dolphin.filter(lambda x: x["has_tran"] == 0)
    dataset_dolphin = dataset_dolphin.remove_columns(
        ["instruction", "input", "output", "has_tran"]
    )

    dataset_iapp = load_dataset("iapp_wiki_qa_squad", split="train")
    dataset_iapp = dataset_iapp.map(re_formate_iappQA, load_from_cache_file=False)
    dataset_iapp = dataset_iapp.remove_columns(
        ["question_id", "article_id", "title", "context", "question", "answers"]
    )

    dataset_thaisum = load_dataset("thaisum", split="train")
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_thaisum)), k=5000
    )
    dataset_thaisum = dataset_thaisum.select(random_test_ids)
    dataset_thaisum = dataset_thaisum.map(
        re_formate_thaisum, load_from_cache_file=False
    )
    dataset_thaisum = dataset_thaisum.remove_columns(
        ["title", "body", "summary", "type", "tags", "url"]
    )

    dataset_xlsum = load_dataset("csebuetnlp/xlsum", "thai", split="train")
    random_test_ids = deterministic_random(42).sample(range(len(dataset_xlsum)), k=5000)
    dataset_xlsum = dataset_xlsum.select(random_test_ids)
    dataset_xlsum = dataset_xlsum.map(re_formate_xlsum, load_from_cache_file=False)
    dataset_xlsum = dataset_xlsum.remove_columns(
        ["id", "url", "title", "summary", "text"]
    )

    dataset_enth = load_dataset("scb_mt_enth_2020", "enth", split="train")

    random_test_ids = deterministic_random(42).sample(range(len(dataset_enth)), k=10000)
    dataset_enth = dataset_enth.select(random_test_ids)
    dataset_then = dataset_enth

    dataset_enth = dataset_enth.map(re_formate_scb_enth, load_from_cache_file=False)
    dataset_enth = dataset_enth.remove_columns(["translation", "subdataset"])

    # dataset_then = load_dataset("scb_mt_enth_2020" , "then",split="train")
    # dataset_then = dataset_then.select(random_test_ids)
    dataset_then = dataset_then.map(re_formate_scb_then, load_from_cache_file=False)
    dataset_then = dataset_then.remove_columns(["translation", "subdataset"])

    # dataset_alpaca_gpt4 = load_dataset("json", data_files="/workspace/sealion/alpaca_gpt4_data.json")
    # dataset_alpaca_gpt4 = dataset_alpaca_gpt4.rename_column("instruction", "Instruction").rename_column("output", "Answer").rename_column("input", "Context")
    # #data_add = add_more_dataset()
    # # print(data_add)
    # print(dataset)
    # print(dataset_dolphin)
    # print(dataset_iapp)
    # print(dataset_thaisum)
    # print(dataset_xlsum)
    # print(dataset_enth)
    # print(dataset_then)
    # print(dataset_alpaca_gpt4)

    dataset_han = load_dataset(
        "csv",
        data_files="/workspace/sealion/examples/pythainlp-han_dataset - pythainlp-han_dataset.csv",
    )
    dataset_han = dataset_han.map(
        re_formate_han, load_from_cache_file=False
    ).remove_columns(["q", "a"])

    all_file = read_file()
    dataset_ = load_dataset("json", data_files=all_file, split="train")
    dataset_1 = dataset_.filter(lambda example: example["split"] == "train")

    dataset_en_th = dataset_1.filter(
        lambda example: example["config"] in ["en_th"]
    )  # ['en_th', 'th', 'thai' ]
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_en_th)), k=40000
    )
    dataset_en_th = dataset_en_th.select(random_test_ids)
    dataset_en_th = dataset_en_th.map(re_formate_xp3, load_from_cache_file=False)
    dataset_en_th = dataset_en_th.remove_columns(
        ["inputs", "targets", "language", "split", "template", "dataset", "config"]
    )

    dataset_thai = dataset_1.filter(lambda example: example["config"] in ["thai"])
    # random_test_ids = deterministic_random(42).sample(range(len(dataset_thai)), k=80000)
    # dataset_thai = dataset_thai.select(random_test_ids)
    dataset_thai = dataset_thai.map(re_formate_xp3, load_from_cache_file=False)
    dataset_thai = dataset_thai.remove_columns(
        ["inputs", "targets", "language", "split", "template", "dataset", "config"]
    )

    dataset_Platypus = load_dataset("garage-bAInd/Open-Platypus", split="train")
    dataset_Platypus = dataset_Platypus.filter(
        lambda example: example["data_source"] != "scienceqa"
        and example["data_source"] != "reclor"
    )  # due to copyright
    dataset_Platypus = dataset_Platypus.map(
        re_formate_platypus, load_from_cache_file=False
    )
    dataset_Platypus = dataset_Platypus.remove_columns(
        ["input", "output", "instruction", "data_source"]
    )

    # dataset_dolly = load_dataset(
    #         "databricks/databricks-dolly-15k",
    #         split="train",
    #         )
    # dataset_dolly = dataset_dolly.map(re_formate_dolly)
    # dataset_dolly = dataset_dolly.remove_columns(['instruction', 'context', 'response', 'category'])

    print(dataset)
    print(dataset_dolphin)
    print(dataset_iapp)
    print(dataset_thaisum)
    print(dataset_xlsum)
    print(dataset_enth)
    print(dataset_then)
    print(dataset_en_th)
    print(dataset_enth)
    print(dataset_thai)
    print(dataset_han["train"])
    print(dataset_Platypus)

    dataset = concatenate_datasets(
        [
            dataset,
            dataset_dolphin,
            dataset_iapp,
            dataset_thaisum,
            dataset_xlsum,
            dataset_enth,
            dataset_then
            # , dataset_alpaca_gpt4["train"],
            ,
            dataset_en_th,
            dataset_thai
            # dataset_dolly,
            ,
            dataset_han["train"],
            dataset_Platypus,
        ]
    )

    dataset = dataset.map(prepare_dolly_text)

    if split_train_test:
        return split_dataset(args, dataset)
    else:
        return dataset


def load_dolly_V7(args, split_train_test=True):
    dataset = load_dataset(
        "pythainlp/final_training_set_v1",
        split="train",
        num_proc=8 if not args.streaming else None,
        streaming=args.streaming,
    )

    dataset = dataset.map(add_prefix)
    dataset = dataset.filter(lambda x: x["bot_morethan_one"] == 2)
    dataset = dataset.remove_columns(
        ["has_context", "bot_morethan_one", "text", "metadata", "nb_token"]
    )

    dataset_dolphin = load_dataset(
        "ehartford/dolphin", data_files="flan1m-alpaca-uncensored.jsonl", split="train"
    )
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_dolphin)), k=10000
    )
    dataset_dolphin = dataset_dolphin.select(random_test_ids)
    dataset_dolphin = dataset_dolphin.map(remove_not_relate_to_thaiEn_dolphin)
    dataset_dolphin = dataset_dolphin.filter(lambda x: x["has_tran"] == 0)
    dataset_dolphin = dataset_dolphin.remove_columns(
        ["instruction", "input", "output", "has_tran"]
    )

    # dataset_iapp = load_dataset("iapp_wiki_qa_squad",split="train")
    # dataset_iapp = dataset_iapp.map(re_formate_iappQA)
    # dataset_iapp = dataset_iapp.remove_columns(['question_id', 'article_id', 'title', 'context', 'question', 'answers'])

    # dataset_thaisum = load_dataset("thaisum",split="train")
    # random_test_ids = deterministic_random(42).sample(range(len(dataset_thaisum)), k=5000)
    # dataset_thaisum = dataset_thaisum.select(random_test_ids)
    # dataset_thaisum  = dataset_thaisum.map(re_formate_thaisum)
    # dataset_thaisum = dataset_thaisum.remove_columns(['title', 'body', 'summary', 'type', 'tags', 'url'])

    # dataset_xlsum = load_dataset("csebuetnlp/xlsum" , "thai",split="train")
    # random_test_ids = deterministic_random(42).sample(range(len(dataset_xlsum)), k=5000)
    # dataset_xlsum = dataset_xlsum.select(random_test_ids)
    # dataset_xlsum = dataset_xlsum.map(re_formate_xlsum)
    # dataset_xlsum = dataset_xlsum.remove_columns(['id', 'url', 'title', 'summary', 'text'])

    # dataset_enth = load_dataset("scb_mt_enth_2020" , "enth",split="train")

    # random_test_ids = deterministic_random(42).sample(range(len(dataset_enth)), k=10000)
    # dataset_enth = dataset_enth.select(random_test_ids)
    # dataset_then = dataset_enth

    # dataset_enth = dataset_enth.map(re_formate_scb_enth)
    # dataset_enth = dataset_enth.remove_columns(['translation', 'subdataset'])

    # #dataset_then = load_dataset("scb_mt_enth_2020" , "then",split="train")
    # #dataset_then = dataset_then.select(random_test_ids)
    # dataset_then  = dataset_then.map(re_formate_scb_then)
    # dataset_then = dataset_then.remove_columns(['translation', 'subdataset'])

    # dataset_alpaca_gpt4 = load_dataset("json", data_files="/workspace/sealion/alpaca_gpt4_data.json")
    # dataset_alpaca_gpt4 = dataset_alpaca_gpt4.rename_column("instruction", "Instruction").rename_column("output", "Answer").rename_column("input", "Context")
    # #data_add = add_more_dataset()
    # # print(data_add)
    # print(dataset)
    # print(dataset_dolphin)
    # print(dataset_iapp)
    # print(dataset_thaisum)
    # print(dataset_xlsum)
    # print(dataset_enth)
    # print(dataset_then)
    # print(dataset_alpaca_gpt4)

    dataset_han = load_dataset(
        "csv",
        data_files="/workspace/sealion/examples/pythainlp-han_dataset - pythainlp-han_dataset.csv",
    )
    dataset_han = dataset_han.map(re_formate_han).remove_columns(["q", "a"])

    # all_file = read_file()
    # dataset_ = load_dataset("json", data_files=all_file, split="train")
    # dataset_1 = dataset_.filter(lambda example: example["split"] == "train")

    # dataset_en_th = dataset_1.filter(lambda example: example["config"] in ['en_th']) # ['en_th', 'th', 'thai' ]
    # random_test_ids = deterministic_random(42).sample(range(len(dataset_en_th)), k=40000)
    # dataset_en_th = dataset_en_th.select(random_test_ids)
    # dataset_en_th = dataset_en_th.map(re_formate_xp3)
    # dataset_en_th = dataset_en_th.remove_columns(['inputs', 'targets', 'language', 'split', 'template', 'dataset', 'config'])

    # dataset_thai = dataset_1.filter(lambda example: example["config"] in ['thai'])
    # # random_test_ids = deterministic_random(42).sample(range(len(dataset_thai)), k=80000)
    # # dataset_thai = dataset_thai.select(random_test_ids)
    # dataset_thai = dataset_thai.map(re_formate_xp3)
    # dataset_thai = dataset_thai.remove_columns(['inputs', 'targets', 'language', 'split', 'template', 'dataset', 'config'])

    dataset_Platypus = load_dataset("garage-bAInd/Open-Platypus", split="train")
    dataset_Platypus = dataset_Platypus.filter(
        lambda example: example["data_source"] != "scienceqa"
        and example["data_source"] != "reclor"
    )  # due to copyright
    dataset_Platypus = dataset_Platypus.map(re_formate_platypus)
    dataset_Platypus = dataset_Platypus.remove_columns(
        ["input", "output", "instruction", "data_source"]
    )

    dataset_tiny = load_dataset("nampdn-ai/tiny-codes", split="train")
    random_test_ids = deterministic_random(42).sample(range(len(dataset_tiny)), k=20000)
    dataset_tiny = dataset_tiny.select(random_test_ids)
    dataset_tiny = dataset_tiny.map(re_formate_code)
    dataset_tiny = dataset_tiny.remove_columns(
        [
            "prompt",
            "main_topic",
            "subtopic",
            "adjective",
            "action_verb",
            "scenario",
            "target_audience",
            "programming_language",
            "common_sense_topic",
            "idx",
            "response",
        ]
    )

    # dataset_dolly = load_dataset(
    #         "databricks/databricks-dolly-15k",
    #         split="train",
    #         )
    # dataset_dolly = dataset_dolly.map(re_formate_dolly)
    # dataset_dolly = dataset_dolly.remove_columns(['instruction', 'context', 'response', 'category'])

    print(dataset)
    print(dataset_dolphin)
    # print(dataset_iapp)
    # print(dataset_thaisum)
    # print(dataset_xlsum)
    # print(dataset_enth)
    # print(dataset_then)
    # print(dataset_en_th)
    # print(dataset_enth)
    # print(dataset_thai)
    print(dataset_han["train"])
    print(dataset_Platypus)
    print(dataset_tiny)

    dataset = concatenate_datasets(
        [
            dataset,
            dataset_dolphin,
            dataset_han["train"],
            dataset_Platypus,
            dataset_tiny,
        ]
    )

    dataset = dataset.map(prepare_dolly_text)
    # print(dataset)

    if split_train_test:
        return split_dataset(args, dataset)
    else:
        return dataset


def load_dolly_V8(args, split_train_test=True):
    dataset = load_dataset(
        "pythainlp/final_training_set_v1",
        split="train",
        num_proc=8 if not args.streaming else None,
        streaming=args.streaming,
    )

    dataset = dataset.map(add_prefix)
    dataset = dataset.filter(lambda x: x["bot_morethan_one"] == 2)
    dataset = dataset.remove_columns(
        ["has_context", "bot_morethan_one", "text", "metadata", "nb_token"]
    )

    dataset_dolphin = load_dataset(
        "ehartford/dolphin", data_files="flan1m-alpaca-uncensored.jsonl", split="train"
    )
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_dolphin)), k=10000
    )
    dataset_dolphin = dataset_dolphin.select(random_test_ids)
    dataset_dolphin = dataset_dolphin.map(remove_not_relate_to_thaiEn_dolphin)
    dataset_dolphin = dataset_dolphin.filter(lambda x: x["has_tran"] == 0)
    dataset_dolphin = dataset_dolphin.remove_columns(
        ["instruction", "input", "output", "has_tran"]
    )

    dataset_iapp = load_dataset("iapp_wiki_qa_squad", split="train")
    dataset_iapp = dataset_iapp.map(re_formate_iappQA)
    dataset_iapp = dataset_iapp.remove_columns(
        ["question_id", "article_id", "title", "context", "question", "answers"]
    )

    dataset_thaisum = load_dataset("thaisum", split="train")
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_thaisum)), k=5000
    )
    dataset_thaisum = dataset_thaisum.select(random_test_ids)
    dataset_thaisum = dataset_thaisum.map(re_formate_thaisum)
    dataset_thaisum = dataset_thaisum.remove_columns(
        ["title", "body", "summary", "type", "tags", "url"]
    )

    dataset_xlsum = load_dataset("csebuetnlp/xlsum", "thai", split="train")
    random_test_ids = deterministic_random(42).sample(range(len(dataset_xlsum)), k=5000)
    dataset_xlsum = dataset_xlsum.select(random_test_ids)
    dataset_xlsum = dataset_xlsum.map(re_formate_xlsum)
    dataset_xlsum = dataset_xlsum.remove_columns(
        ["id", "url", "title", "summary", "text"]
    )

    dataset_enth = load_dataset("scb_mt_enth_2020", "enth", split="train")

    random_test_ids = deterministic_random(42).sample(range(len(dataset_enth)), k=5000)
    dataset_enth = dataset_enth.select(random_test_ids)
    dataset_then = dataset_enth

    dataset_enth = dataset_enth.map(re_formate_scb_enth)
    dataset_enth = dataset_enth.remove_columns(["translation", "subdataset"])

    # dataset_then = load_dataset("scb_mt_enth_2020" , "then",split="train")
    # dataset_then = dataset_then.select(random_test_ids)
    dataset_then = dataset_then.map(re_formate_scb_then)
    dataset_then = dataset_then.remove_columns(["translation", "subdataset"])

    # dataset_alpaca_gpt4 = load_dataset("json", data_files="/workspace/sealion/alpaca_gpt4_data.json")
    # dataset_alpaca_gpt4 = dataset_alpaca_gpt4.rename_column("instruction", "Instruction").rename_column("output", "Answer").rename_column("input", "Context")
    # #data_add = add_more_dataset()
    # # print(data_add)
    # print(dataset)
    # print(dataset_dolphin)
    # print(dataset_iapp)
    # print(dataset_thaisum)
    # print(dataset_xlsum)
    # print(dataset_enth)
    # print(dataset_then)
    # print(dataset_alpaca_gpt4)

    dataset_han = load_dataset(
        "csv",
        data_files="/workspace/sealion/examples/pythainlp-han_dataset - pythainlp-han_dataset.csv",
    )
    dataset_han = dataset_han.map(re_formate_han).remove_columns(["q", "a"])

    all_file = read_file()
    dataset_ = load_dataset("json", data_files=all_file, split="train")
    dataset_1 = dataset_.filter(lambda example: example["split"] == "train")

    dataset_en_th = dataset_1.filter(
        lambda example: example["config"] in ["en_th"]
    )  # ['en_th', 'th', 'thai' ]
    random_test_ids = deterministic_random(42).sample(range(len(dataset_en_th)), k=5000)
    dataset_en_th = dataset_en_th.select(random_test_ids)
    dataset_en_th = dataset_en_th.map(re_formate_xp3)
    dataset_en_th = dataset_en_th.remove_columns(
        ["inputs", "targets", "language", "split", "template", "dataset", "config"]
    )

    dataset_thai = dataset_1.filter(lambda example: example["config"] in ["thai"])
    random_test_ids = deterministic_random(42).sample(range(len(dataset_thai)), k=10000)
    dataset_thai = dataset_thai.select(random_test_ids)
    dataset_thai = dataset_thai.map(re_formate_xp3)
    dataset_thai = dataset_thai.remove_columns(
        ["inputs", "targets", "language", "split", "template", "dataset", "config"]
    )

    dataset_Platypus = load_dataset("garage-bAInd/Open-Platypus", split="train")
    dataset_Platypus = dataset_Platypus.filter(
        lambda example: example["data_source"] != "scienceqa"
        and example["data_source"] != "reclor"
    )  # due to copyright
    dataset_Platypus = dataset_Platypus.map(re_formate_platypus)
    dataset_Platypus = dataset_Platypus.remove_columns(
        ["input", "output", "instruction", "data_source"]
    )

    dataset_tiny = load_dataset("nampdn-ai/tiny-codes", split="train")
    random_test_ids = deterministic_random(42).sample(range(len(dataset_tiny)), k=20000)
    dataset_tiny = dataset_tiny.select(random_test_ids)
    dataset_tiny = dataset_tiny.map(re_formate_code)
    dataset_tiny = dataset_tiny.remove_columns(
        [
            "prompt",
            "main_topic",
            "subtopic",
            "adjective",
            "action_verb",
            "scenario",
            "target_audience",
            "programming_language",
            "common_sense_topic",
            "idx",
            "response",
        ]
    )

    # dataset_dolly = load_dataset(
    #         "databricks/databricks-dolly-15k",
    #         split="train",
    #         )
    # dataset_dolly = dataset_dolly.map(re_formate_dolly)
    # dataset_dolly = dataset_dolly.remove_columns(['instruction', 'context', 'response', 'category'])

    print(dataset)
    print(dataset_dolphin)
    print(dataset_iapp)
    print(dataset_thaisum)
    print(dataset_xlsum)
    print(dataset_enth)
    print(dataset_then)
    print(dataset_en_th)
    print(dataset_thai)
    print(dataset_han["train"])
    print(dataset_Platypus)
    print(dataset_tiny)

    dataset = concatenate_datasets(
        [
            dataset,
            dataset_dolphin,
            dataset_iapp,
            dataset_thaisum,
            dataset_xlsum,
            dataset_enth,
            dataset_then,
            dataset_en_th,
            dataset_thai,
            dataset_han["train"],
            dataset_Platypus,
            dataset_tiny,
        ]
    )

    dataset = dataset.map(prepare_dolly_text)
    # print(dataset)

    if split_train_test:
        return split_dataset(args, dataset)
    else:
        return dataset


def load_dolly_V9(args, split_train_test=True):
    dataset = load_dataset(
        "pythainlp/final_training_set_v1",
        split="train",
        num_proc=8 if not args.streaming else None,
        streaming=args.streaming,
    )

    dataset = dataset.map(add_prefix)
    dataset = dataset.filter(lambda x: x["bot_morethan_one"] == 2)
    dataset = dataset.remove_columns(
        ["has_context", "bot_morethan_one", "text", "metadata", "nb_token"]
    )

    # dataset_dolphin = load_dataset("ehartford/dolphin",data_files="flan1m-alpaca-uncensored.jsonl" ,split="train")
    # random_test_ids = deterministic_random(42).sample(range(len(dataset_dolphin)), k=10000)
    # dataset_dolphin = dataset_dolphin.select(random_test_ids)
    # dataset_dolphin = dataset_dolphin.map(remove_not_relate_to_thaiEn_dolphin)
    # dataset_dolphin = dataset_dolphin.filter(lambda x: x["has_tran"] == 0)
    # dataset_dolphin = dataset_dolphin.remove_columns(['instruction', 'input', 'output', 'has_tran'])

    dataset_iapp = load_dataset("iapp_wiki_qa_squad", split="train")
    dataset_iapp = dataset_iapp.map(re_formate_iappQA)
    dataset_iapp = dataset_iapp.remove_columns(
        ["question_id", "article_id", "title", "context", "question", "answers"]
    )

    dataset_thaisum = load_dataset("thaisum", split="train")
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_thaisum)), k=5000
    )
    dataset_thaisum = dataset_thaisum.select(random_test_ids)
    dataset_thaisum = dataset_thaisum.map(re_formate_thaisum)
    dataset_thaisum = dataset_thaisum.remove_columns(
        ["title", "body", "summary", "type", "tags", "url"]
    )

    dataset_xlsum = load_dataset("csebuetnlp/xlsum", "thai", split="train")
    random_test_ids = deterministic_random(42).sample(range(len(dataset_xlsum)), k=5000)
    dataset_xlsum = dataset_xlsum.select(random_test_ids)
    dataset_xlsum = dataset_xlsum.map(re_formate_xlsum)
    dataset_xlsum = dataset_xlsum.remove_columns(
        ["id", "url", "title", "summary", "text"]
    )

    dataset_enth = load_dataset("scb_mt_enth_2020", "enth", split="train")

    random_test_ids = deterministic_random(42).sample(range(len(dataset_enth)), k=10000)
    dataset_enth = dataset_enth.select(random_test_ids)
    dataset_then = dataset_enth

    dataset_enth = dataset_enth.map(re_formate_scb_enth)
    dataset_enth = dataset_enth.remove_columns(["translation", "subdataset"])

    # dataset_then = load_dataset("scb_mt_enth_2020" , "then",split="train")
    # dataset_then = dataset_then.select(random_test_ids)
    dataset_then = dataset_then.map(re_formate_scb_then)
    dataset_then = dataset_then.remove_columns(["translation", "subdataset"])

    # dataset_alpaca_gpt4 = load_dataset("json", data_files="/workspace/sealion/alpaca_gpt4_data.json")
    # dataset_alpaca_gpt4 = dataset_alpaca_gpt4.rename_column("instruction", "Instruction").rename_column("output", "Answer").rename_column("input", "Context")
    # #data_add = add_more_dataset()
    # # print(data_add)
    # print(dataset)
    # print(dataset_dolphin)
    # print(dataset_iapp)
    # print(dataset_thaisum)
    # print(dataset_xlsum)
    # print(dataset_enth)
    # print(dataset_then)
    # print(dataset_alpaca_gpt4)

    dataset_han = load_dataset(
        "csv",
        data_files="/workspace/sealion/examples/pythainlp-han_dataset - pythainlp-han_dataset.csv",
    )
    dataset_han = dataset_han.map(re_formate_han).remove_columns(["q", "a"])

    # all_file = read_file()
    # dataset_ = load_dataset("json", data_files=all_file, split="train")
    # dataset_1 = dataset_.filter(lambda example: example["split"] == "train")

    # dataset_en_th = dataset_1.filter(lambda example: example["config"] in ['en_th']) # ['en_th', 'th', 'thai' ]
    # random_test_ids = deterministic_random(42).sample(range(len(dataset_en_th)), k=5000)
    # dataset_en_th = dataset_en_th.select(random_test_ids)
    # dataset_en_th = dataset_en_th.map(re_formate_xp3)
    # dataset_en_th = dataset_en_th.remove_columns(['inputs', 'targets', 'language', 'split', 'template', 'dataset', 'config'])

    # dataset_thai = dataset_1.filter(lambda example: example["config"] in ['thai'])
    # random_test_ids = deterministic_random(42).sample(range(len(dataset_thai)), k=10000)
    # dataset_thai = dataset_thai.select(random_test_ids)
    # dataset_thai = dataset_thai.map(re_formate_xp3)
    # dataset_thai = dataset_thai.remove_columns(['inputs', 'targets', 'language', 'split', 'template', 'dataset', 'config'])

    dataset_Platypus = load_dataset("garage-bAInd/Open-Platypus", split="train")
    dataset_Platypus = dataset_Platypus.filter(
        lambda example: example["data_source"] != "scienceqa"
        and example["data_source"] != "reclor"
    )  # due to copyright
    dataset_Platypus = dataset_Platypus.map(re_formate_platypus)
    dataset_Platypus = dataset_Platypus.remove_columns(
        ["input", "output", "instruction", "data_source"]
    )

    dataset_tiny = load_dataset("nampdn-ai/tiny-codes", split="train")
    random_test_ids = deterministic_random(42).sample(range(len(dataset_tiny)), k=20000)
    dataset_tiny = dataset_tiny.select(random_test_ids)
    dataset_tiny = dataset_tiny.map(re_formate_code)
    dataset_tiny = dataset_tiny.remove_columns(
        [
            "prompt",
            "main_topic",
            "subtopic",
            "adjective",
            "action_verb",
            "scenario",
            "target_audience",
            "programming_language",
            "common_sense_topic",
            "idx",
            "response",
        ]
    )

    dataset_cot = load_dataset("kaist-ai/CoT-Collection", split="train")
    random_test_ids = deterministic_random(42).sample(range(len(dataset_cot)), k=60000)
    dataset_cot = dataset_cot.select(random_test_ids)
    dataset_cot = dataset_cot.map(re_formate_cot)
    dataset_cot = dataset_cot.remove_columns(
        ["source", "target", "rationale", "task", "type"]
    )

    # dataset_dolly = load_dataset(
    #         "databricks/databricks-dolly-15k",
    #         split="train",
    #         )
    # dataset_dolly = dataset_dolly.map(re_formate_dolly)
    # dataset_dolly = dataset_dolly.remove_columns(['instruction', 'context', 'response', 'category'])

    print(dataset)
    # print(dataset_dolphin)
    print(dataset_iapp)
    print(dataset_thaisum)
    print(dataset_xlsum)
    print(dataset_enth)
    print(dataset_then)
    # print(dataset_en_th)
    # print(dataset_thai)
    print(dataset_han["train"])
    print(dataset_Platypus)
    print(dataset_tiny)
    # print(dataset_dolly)
    print(dataset_cot)

    dataset = concatenate_datasets(
        [
            dataset,
            # dataset_dolphin,
            dataset_iapp,
            dataset_thaisum,
            dataset_xlsum,
            dataset_enth,
            dataset_then,
            # dataset_en_th,
            # dataset_thai,
            dataset_han["train"],
            dataset_Platypus,
            dataset_tiny,
            # dataset_dolly,
            dataset_cot,
        ]
    )
    dataset = dataset.map(prepare_dolly_text)
    # print(dataset)

    if split_train_test:
        return split_dataset(args, dataset)
    else:
        return dataset


def load_dolly_V19(
    args, split_train_test=True
):  # V17,V19 lora 512,256 V17(prompt \n),V19 (###USER ...)
    dataset = load_dataset(
        "pythainlp/final_training_set_v1",
        split="train",
        num_proc=8 if not args.streaming else None,
        streaming=args.streaming,
    )
    # dataset = load_from_disk("/project/lt200061-changx/finetune_SFT/loaded_datasets/final_training_set_v1")

    dataset = dataset.map(add_prefix)
    dataset = dataset.filter(lambda x: x["bot_morethan_one"] == 2)
    dataset = dataset.remove_columns(
        ["has_context", "bot_morethan_one", "text", "metadata", "nb_token"]
    )

    dataset_dolphin = load_dataset(
        "ehartford/dolphin", data_files="flan1m-alpaca-uncensored.jsonl", split="train"
    )
    # dataset_dolphin = load_from_disk("/project/lt200061-changx/finetune_SFT/loaded_datasets/dolphin")
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_dolphin)), k=10000
    )
    dataset_dolphin = dataset_dolphin.select(random_test_ids)
    dataset_dolphin = dataset_dolphin.map(remove_not_relate_to_thaiEn_dolphin)
    dataset_dolphin = dataset_dolphin.filter(lambda x: x["has_tran"] == 0)
    dataset_dolphin = dataset_dolphin.remove_columns(
        ["instruction", "input", "output", "has_tran"]
    )

    dataset_iapp = load_dataset("iapp_wiki_qa_squad", split="train")
    # dataset_iapp = load_from_disk("/project/lt200061-changx/finetune_SFT/loaded_datasets/iapp_wiki_qa_squad")
    dataset_iapp = dataset_iapp.map(re_formate_iappQA)
    dataset_iapp = dataset_iapp.remove_columns(
        ["question_id", "article_id", "title", "context", "question", "answers"]
    )

    dataset_thaisum = load_dataset("thaisum", split="train")
    # dataset_thaisum= load_from_disk("/project/lt200061-changx/finetune_SFT/loaded_datasets/thaisum")
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_thaisum)), k=5000
    )
    dataset_thaisum = dataset_thaisum.select(random_test_ids)
    dataset_thaisum = dataset_thaisum.map(re_formate_thaisum)
    dataset_thaisum = dataset_thaisum.remove_columns(
        ["title", "body", "summary", "type", "tags", "url"]
    )

    dataset_xlsum = load_dataset("csebuetnlp/xlsum", "thai", split="train")
    # dataset_xlsum = load_from_disk("/project/lt200061-changx/finetune_SFT/loaded_datasets/xlsum")
    random_test_ids = deterministic_random(42).sample(range(len(dataset_xlsum)), k=5000)
    dataset_xlsum = dataset_xlsum.select(random_test_ids)
    dataset_xlsum = dataset_xlsum.map(re_formate_xlsum)
    dataset_xlsum = dataset_xlsum.remove_columns(
        ["id", "url", "title", "summary", "text"]
    )

    dataset_enth = load_dataset("scb_mt_enth_2020", "enth", split="train")
    # dataset_enth = load_from_disk("/project/lt200061-changx/finetune_SFT/loaded_datasets/scb_mt_enth_2020")

    random_test_ids = deterministic_random(42).sample(range(len(dataset_enth)), k=10000)
    dataset_enth = dataset_enth.select(random_test_ids)
    dataset_then = dataset_enth

    dataset_enth = dataset_enth.map(re_formate_scb_enth)
    dataset_enth = dataset_enth.remove_columns(["translation", "subdataset"])

    # dataset_then = load_dataset("scb_mt_enth_2020" , "then",split="train")
    # dataset_then = dataset_then.select(random_test_ids)
    dataset_then = dataset_then.map(re_formate_scb_then)
    dataset_then = dataset_then.remove_columns(["translation", "subdataset"])

    # dataset_alpaca_gpt4 = load_dataset("json", data_files="/workspace/sealion/alpaca_gpt4_data.json")
    # dataset_alpaca_gpt4 = dataset_alpaca_gpt4.rename_column("instruction", "Instruction").rename_column("output", "Answer").rename_column("input", "Context")
    # #data_add = add_more_dataset()
    # # print(data_add)
    # print(dataset)
    # print(dataset_dolphin)
    # print(dataset_iapp)
    # print(dataset_thaisum)
    # print(dataset_xlsum)
    # print(dataset_enth)
    # print(dataset_then)
    # print(dataset_alpaca_gpt4)

    dataset_han = load_dataset(
        "csv",
        data_files="/workspace/sealion/examples/pythainlp-han_dataset - pythainlp-han_dataset.csv",
    )
    dataset_han = dataset_han.map(re_formate_han).remove_columns(["q", "a"])

    all_file = read_file()
    dataset_ = load_dataset("json", data_files=all_file, split="train")
    dataset_1 = dataset_.filter(lambda example: example["split"] == "train")

    dataset_en_th = dataset_1.filter(
        lambda example: example["config"] in ["en_th"]
    )  # ['en_th', 'th', 'thai' ]
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_en_th)), k=40000
    )
    dataset_en_th = dataset_en_th.select(random_test_ids)
    dataset_en_th = dataset_en_th.map(re_formate_xp3)
    dataset_en_th = dataset_en_th.remove_columns(
        ["inputs", "targets", "language", "split", "template", "dataset", "config"]
    )

    dataset_thai = dataset_1.filter(lambda example: example["config"] in ["thai"])
    # random_test_ids = deterministic_random(42).sample(range(len(dataset_thai)), k=80000)
    # dataset_thai = dataset_thai.select(random_test_ids)
    dataset_thai = dataset_thai.map(re_formate_xp3)
    dataset_thai = dataset_thai.remove_columns(
        ["inputs", "targets", "language", "split", "template", "dataset", "config"]
    )

    dataset_Platypus = load_dataset("garage-bAInd/Open-Platypus", split="train")
    # dataset_Platypus = load_from_disk("/project/lt200061-changx/finetune_SFT/loaded_datasets/Open-Platypus")
    dataset_Platypus = dataset_Platypus.filter(
        lambda example: example["data_source"] != "scienceqa"
        and example["data_source"] != "reclor"
    )  # due to copyright
    dataset_Platypus = dataset_Platypus.map(re_formate_platypus)
    dataset_Platypus = dataset_Platypus.remove_columns(
        ["input", "output", "instruction", "data_source"]
    )

    # dataset_dolly = load_dataset(
    #         "databricks/databricks-dolly-15k",
    #         split="train",
    #         )
    # dataset_dolly = dataset_dolly.map(re_formate_dolly)
    # dataset_dolly = dataset_dolly.remove_columns(['instruction', 'context', 'response', 'category'])

    print(dataset)
    print(dataset_dolphin)
    print(dataset_iapp)
    print(dataset_thaisum)
    print(dataset_xlsum)
    print(dataset_enth)
    print(dataset_then)
    print(dataset_en_th)
    print(dataset_enth)
    print(dataset_thai)
    print(dataset_han["train"])
    print(dataset_Platypus)

    dataset = concatenate_datasets(
        [
            dataset,
            dataset_dolphin,
            dataset_iapp,
            dataset_thaisum,
            dataset_xlsum,
            dataset_enth,
            dataset_then
            # , dataset_alpaca_gpt4["train"],
            ,
            dataset_en_th,
            dataset_thai
            # dataset_dolly,
            ,
            dataset_han["train"],
            dataset_Platypus,
        ]
    )

    dataset = dataset.map(prepare_dolly_text)
    # print(dataset)

    if split_train_test:
        return split_dataset(args, dataset)
    else:
        return dataset


def load_dolly_V21(args, split_train_test=True):  # V21
    dataset = load_dataset(
        "pythainlp/final_training_set_v1",
        split="train",
        num_proc=8 if not args.streaming else None,
        streaming=args.streaming,
    )
    # dataset = load_from_disk("/project/lt200061-changx/finetune_SFT/loaded_datasets/final_training_set_v1")

    random_test_ids = deterministic_random(42).sample(range(len(dataset)), k=300000)
    dataset = dataset.select(random_test_ids)
    dataset = dataset.map(add_prefix, load_from_cache_file=False)
    dataset = dataset.filter(lambda x: x["bot_morethan_one"] == 2)
    dataset = dataset.remove_columns(
        ["has_context", "bot_morethan_one", "text", "metadata", "nb_token"]
    )

    # dataset_dolphin = load_dataset("ehartford/dolphin",data_files="flan1m-alpaca-uncensored.jsonl" ,split="train")
    # random_test_ids = deterministic_random(42).sample(range(len(dataset_dolphin)), k=100000)
    # dataset_dolphin = dataset_dolphin.select(random_test_ids)
    # dataset_dolphin = dataset_dolphin.map(remove_not_relate_to_thaiEn_dolphin, load_from_cache_file=False)
    # dataset_dolphin = dataset_dolphin.filter(lambda x: x["has_tran"] == 0)
    # dataset_dolphin = dataset_dolphin.remove_columns(['instruction', 'input', 'output', 'has_tran'])

    dataset_iapp = load_dataset("iapp_wiki_qa_squad", split="train")
    # dataset_iapp = load_from_disk("/project/lt200061-changx/finetune_SFT/loaded_datasets/iapp_wiki_qa_squad")
    dataset_iapp = dataset_iapp.map(re_formate_iappQA, load_from_cache_file=False)
    dataset_iapp = dataset_iapp.remove_columns(
        ["question_id", "article_id", "title", "context", "question", "answers"]
    )

    dataset_thaisum = load_dataset("thaisum", split="train")
    # dataset_thaisum= load_from_disk("/project/lt200061-changx/finetune_SFT/loaded_datasets/thaisum")
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_thaisum)), k=5000
    )
    dataset_thaisum = dataset_thaisum.select(random_test_ids)
    dataset_thaisum = dataset_thaisum.map(
        re_formate_thaisum, load_from_cache_file=False
    )
    dataset_thaisum = dataset_thaisum.remove_columns(
        ["title", "body", "summary", "type", "tags", "url"]
    )

    dataset_xlsum = load_dataset("csebuetnlp/xlsum", "thai", split="train")
    # dataset_xlsum = load_from_disk("/project/lt200061-changx/finetune_SFT/loaded_datasets/xlsum")
    random_test_ids = deterministic_random(42).sample(range(len(dataset_xlsum)), k=5000)
    dataset_xlsum = dataset_xlsum.select(random_test_ids)
    dataset_xlsum = dataset_xlsum.map(re_formate_xlsum, load_from_cache_file=False)
    dataset_xlsum = dataset_xlsum.remove_columns(
        ["id", "url", "title", "summary", "text"]
    )

    dataset_enth = load_dataset("scb_mt_enth_2020", "enth", split="train")
    # dataset_enth = load_from_disk("/project/lt200061-changx/finetune_SFT/loaded_datasets/scb_mt_enth_2020")

    random_test_ids = deterministic_random(42).sample(range(len(dataset_enth)), k=10000)
    dataset_enth = dataset_enth.select(random_test_ids)
    dataset_then = dataset_enth

    dataset_enth = dataset_enth.map(re_formate_scb_enth, load_from_cache_file=False)
    dataset_enth = dataset_enth.remove_columns(["translation", "subdataset"])

    # dataset_then = load_dataset("scb_mt_enth_2020" , "then",split="train")
    # dataset_then = dataset_then.select(random_test_ids)
    dataset_then = dataset_then.map(re_formate_scb_then, load_from_cache_file=False)
    dataset_then = dataset_then.remove_columns(["translation", "subdataset"])

    # dataset_alpaca_gpt4 = load_dataset("json", data_files="/workspace/sealion/alpaca_gpt4_data.json")
    # dataset_alpaca_gpt4 = dataset_alpaca_gpt4.rename_column("instruction", "Instruction").rename_column("output", "Answer").rename_column("input", "Context")
    # #data_add = add_more_dataset()
    # # print(data_add)
    # print(dataset)
    # print(dataset_dolphin)
    # print(dataset_iapp)
    # print(dataset_thaisum)
    # print(dataset_xlsum)
    # print(dataset_enth)
    # print(dataset_then)
    # print(dataset_alpaca_gpt4)

    dataset_han = load_dataset(
        "csv",
        data_files="/workspace/sealion/examples/pythainlp-han_dataset - pythainlp-han_dataset.csv",
    )
    dataset_han = dataset_han.map(
        re_formate_han, load_from_cache_file=False
    ).remove_columns(["q", "a"])

    all_file = read_file()
    dataset_ = load_dataset("json", data_files=all_file, split="train")
    dataset_1 = dataset_.filter(lambda example: example["split"] == "train")

    dataset_en_th = dataset_1.filter(
        lambda example: example["config"] in ["en_th"]
    )  # ['en_th', 'th', 'thai' ]
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_en_th)), k=40000
    )
    dataset_en_th = dataset_en_th.select(random_test_ids)
    dataset_en_th = dataset_en_th.map(re_formate_xp3, load_from_cache_file=False)
    dataset_en_th = dataset_en_th.remove_columns(
        ["inputs", "targets", "language", "split", "template", "dataset", "config"]
    )

    dataset_thai = dataset_1.filter(lambda example: example["config"] in ["thai"])
    # random_test_ids = deterministic_random(42).sample(range(len(dataset_thai)), k=80000)
    # dataset_thai = dataset_thai.select(random_test_ids)
    dataset_thai = dataset_thai.map(re_formate_xp3, load_from_cache_file=False)
    dataset_thai = dataset_thai.remove_columns(
        ["inputs", "targets", "language", "split", "template", "dataset", "config"]
    )

    dataset_Platypus = load_dataset("garage-bAInd/Open-Platypus", split="train")
    # dataset_Platypus = load_from_disk("/project/lt200061-changx/finetune_SFT/loaded_datasets/Open-Platypus")
    dataset_Platypus = dataset_Platypus.filter(
        lambda example: example["data_source"] != "scienceqa"
        and example["data_source"] != "reclor"
    )  # due to copyright
    dataset_Platypus = dataset_Platypus.map(
        re_formate_platypus, load_from_cache_file=False
    )
    dataset_Platypus = dataset_Platypus.remove_columns(
        ["input", "output", "instruction", "data_source"]
    )

    # dataset_dolly = load_dataset(
    #         "databricks/databricks-dolly-15k",
    #         split="train",
    #         )
    # dataset_dolly = dataset_dolly.map(re_formate_dolly)
    # dataset_dolly = dataset_dolly.remove_columns(['instruction', 'context', 'response', 'category'])

    dataset_tiny = load_dataset("nampdn-ai/tiny-codes", split="train")
    random_test_ids = deterministic_random(42).sample(range(len(dataset_tiny)), k=50000)
    dataset_tiny = dataset_tiny.select(random_test_ids)
    dataset_tiny = dataset_tiny.map(re_formate_code, load_from_cache_file=False)
    dataset_tiny = dataset_tiny.remove_columns(
        [
            "prompt",
            "main_topic",
            "subtopic",
            "adjective",
            "action_verb",
            "scenario",
            "target_audience",
            "programming_language",
            "common_sense_topic",
            "idx",
            "response",
        ]
    )

    dataset_cot = load_dataset("kaist-ai/CoT-Collection", split="train")
    random_test_ids = deterministic_random(42).sample(range(len(dataset_cot)), k=100000)
    dataset_cot = dataset_cot.select(random_test_ids)
    dataset_cot = dataset_cot.map(re_formate_cot, load_from_cache_file=False)
    dataset_cot = dataset_cot.remove_columns(
        ["source", "target", "rationale", "task", "type"]
    )

    dataset_alpaca_gpt4 = load_dataset(
        "json", data_files="/workspace/sealion/alpaca_gpt4_data.json"
    )
    dataset_alpaca_gpt4 = (
        dataset_alpaca_gpt4.rename_column("instruction", "Instruction")
        .rename_column("output", "Answer")
        .rename_column("input", "Context")
    )

    print(dataset)
    # print(dataset_dolphin)
    print(dataset_iapp)
    print(dataset_thaisum)
    print(dataset_xlsum)
    print(dataset_enth)
    print(dataset_then)
    print(dataset_en_th)
    print(dataset_enth)
    print(dataset_thai)
    print(dataset_han["train"])
    print(dataset_Platypus)
    print(dataset_tiny)
    print(dataset_cot)
    print(dataset_alpaca_gpt4["train"])

    dataset = concatenate_datasets(
        [
            dataset
            # ,dataset_dolphin
            ,
            dataset_iapp,
            dataset_thaisum,
            dataset_xlsum,
            dataset_enth,
            dataset_then,
            dataset_alpaca_gpt4["train"],
            dataset_en_th,
            dataset_thai
            # dataset_dolly,
            ,
            dataset_han["train"],
            dataset_Platypus,
            dataset_tiny,
            dataset_cot,
        ]
    )

    dataset = dataset.map(prepare_dolly_text)
    # print(dataset)

    if split_train_test:
        return split_dataset(args, dataset)
    else:
        return dataset


def load_dolly_V10(args, split_train_test=True):
    dataset = load_dataset(
        "pythainlp/final_training_set_v1",
        split="train",
        num_proc=8 if not args.streaming else None,
        streaming=args.streaming,
    )

    dataset = dataset.map(add_prefix)
    dataset = dataset.filter(lambda x: x["bot_morethan_one"] == 2)
    dataset = dataset.remove_columns(
        ["has_context", "bot_morethan_one", "text", "metadata", "nb_token"]
    )

    dataset_dolphin = load_dataset(
        "ehartford/dolphin", data_files="flan1m-alpaca-uncensored.jsonl", split="train"
    )
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_dolphin)), k=10000
    )
    dataset_dolphin = dataset_dolphin.select(random_test_ids)
    dataset_dolphin = dataset_dolphin.map(remove_not_relate_to_thaiEn_dolphin)
    dataset_dolphin = dataset_dolphin.filter(lambda x: x["has_tran"] == 0)
    dataset_dolphin = dataset_dolphin.remove_columns(
        ["instruction", "input", "output", "has_tran"]
    )

    dataset_iapp = load_dataset("iapp_wiki_qa_squad", split="train")
    dataset_iapp = dataset_iapp.map(re_formate_iappQA)
    dataset_iapp = dataset_iapp.remove_columns(
        ["question_id", "article_id", "title", "context", "question", "answers"]
    )

    dataset_thaisum = load_dataset("thaisum", split="train")
    # random_test_ids = deterministic_random(42).sample(range(len(dataset_thaisum)), k=5000)
    # dataset_thaisum = dataset_thaisum.select(random_test_ids)
    dataset_thaisum = dataset_thaisum.map(re_formate_thaisum)
    dataset_thaisum = dataset_thaisum.remove_columns(
        ["title", "body", "summary", "type", "tags", "url"]
    )

    dataset_xlsum = load_dataset("csebuetnlp/xlsum", "thai", split="train")
    # random_test_ids = deterministic_random(42).sample(range(len(dataset_xlsum)), k=5000)
    # dataset_xlsum = dataset_xlsum.select(random_test_ids)
    dataset_xlsum = dataset_xlsum.map(re_formate_xlsum)
    dataset_xlsum = dataset_xlsum.remove_columns(
        ["id", "url", "title", "summary", "text"]
    )

    dataset_enth = load_dataset("scb_mt_enth_2020", "enth", split="train")

    random_test_ids = deterministic_random(42).sample(range(len(dataset_enth)), k=50000)
    dataset_enth = dataset_enth.select(random_test_ids)
    dataset_then = dataset_enth

    dataset_enth = dataset_enth.map(re_formate_scb_enth)
    dataset_enth = dataset_enth.remove_columns(["translation", "subdataset"])

    # dataset_then = load_dataset("scb_mt_enth_2020" , "then",split="train")
    # dataset_then = dataset_then.select(random_test_ids)
    dataset_then = dataset_then.map(re_formate_scb_then)
    dataset_then = dataset_then.remove_columns(["translation", "subdataset"])

    dataset_alpaca_gpt4 = load_dataset(
        "json", data_files="/workspace/sealion/alpaca_gpt4_data.json"
    )
    dataset_alpaca_gpt4 = (
        dataset_alpaca_gpt4.rename_column("instruction", "Instruction")
        .rename_column("output", "Answer")
        .rename_column("input", "Context")
    )

    dataset_han = load_dataset(
        "csv",
        data_files="/workspace/sealion/examples/pythainlp-han_dataset - pythainlp-han_dataset.csv",
    )
    dataset_han = dataset_han.map(re_formate_han).remove_columns(["q", "a"])

    all_file = read_file()
    dataset_ = load_dataset("json", data_files=all_file, split="train")
    dataset_1 = dataset_.filter(lambda example: example["split"] == "train")

    dataset_en_th = dataset_1.filter(
        lambda example: example["config"] in ["en_th"]
    )  # ['en_th', 'th', 'thai' ]
    # random_test_ids = deterministic_random(42).sample(range(len(dataset_en_th)), k=5000)
    # dataset_en_th = dataset_en_th.select(random_test_ids)
    dataset_en_th = dataset_en_th.map(re_formate_xp3)
    dataset_en_th = dataset_en_th.remove_columns(
        ["inputs", "targets", "language", "split", "template", "dataset", "config"]
    )

    dataset_thai = dataset_1.filter(lambda example: example["config"] in ["thai"])
    # random_test_ids = deterministic_random(42).sample(range(len(dataset_thai)), k=10000)
    # dataset_thai = dataset_thai.select(random_test_ids)
    dataset_thai = dataset_thai.map(re_formate_xp3)
    dataset_thai = dataset_thai.remove_columns(
        ["inputs", "targets", "language", "split", "template", "dataset", "config"]
    )

    dataset_Platypus = load_dataset("garage-bAInd/Open-Platypus", split="train")
    dataset_Platypus = dataset_Platypus.filter(
        lambda example: example["data_source"] != "scienceqa"
        and example["data_source"] != "reclor"
    )  # due to copyright
    dataset_Platypus = dataset_Platypus.map(re_formate_platypus)
    dataset_Platypus = dataset_Platypus.remove_columns(
        ["input", "output", "instruction", "data_source"]
    )

    dataset_tiny = load_dataset("nampdn-ai/tiny-codes", split="train")
    random_test_ids = deterministic_random(42).sample(range(len(dataset_tiny)), k=20000)
    dataset_tiny = dataset_tiny.select(random_test_ids)
    dataset_tiny = dataset_tiny.map(re_formate_code)
    dataset_tiny = dataset_tiny.remove_columns(
        [
            "prompt",
            "main_topic",
            "subtopic",
            "adjective",
            "action_verb",
            "scenario",
            "target_audience",
            "programming_language",
            "common_sense_topic",
            "idx",
            "response",
        ]
    )

    dataset_cot = load_dataset("kaist-ai/CoT-Collection", split="train")
    random_test_ids = deterministic_random(42).sample(range(len(dataset_cot)), k=60000)
    dataset_cot = dataset_cot.select(random_test_ids)
    dataset_cot = dataset_cot.map(re_formate_cot)
    dataset_cot = dataset_cot.remove_columns(
        ["source", "target", "rationale", "task", "type"]
    )

    # dataset_dolly = load_dataset(
    #         "databricks/databricks-dolly-15k",
    #         split="train",
    #         )
    # dataset_dolly = dataset_dolly.map(re_formate_dolly)
    # dataset_dolly = dataset_dolly.remove_columns(['instruction', 'context', 'response', 'category'])

    print(dataset)
    # print(dataset_dolphin)
    print(dataset_iapp)
    print(dataset_thaisum)
    print(dataset_xlsum)
    print(dataset_enth)
    print(dataset_then)
    print(dataset_en_th)
    print(dataset_thai)
    print(dataset_alpaca_gpt4["train"])
    print(dataset_han["train"])
    print(dataset_Platypus)
    print(dataset_tiny)
    # print(dataset_dolly)
    print(dataset_cot)

    dataset = concatenate_datasets(
        [
            dataset,
            # dataset_dolphin,
            dataset_iapp,
            dataset_thaisum,
            dataset_xlsum,
            dataset_enth,
            dataset_then,
            dataset_en_th,
            dataset_thai,
            dataset_alpaca_gpt4["train"],
            dataset_han["train"],
            dataset_Platypus,
            dataset_tiny,
            # dataset_dolly,
            dataset_cot,
        ]
    )
    dataset = dataset.map(prepare_dolly_text)
    # print(dataset)

    if split_train_test:
        return split_dataset(args, dataset)
    else:
        return dataset


def load_dolly_V11(args, split_train_test=True):
    # dataset = load_dataset(
    #         "pythainlp/final_training_set_v1",
    #         split="train",
    #         num_proc=8 if not args.streaming else None,
    #         streaming = args.streaming
    #         )

    # dataset = dataset.map(add_prefix)
    # dataset = dataset.filter(lambda x: x["bot_morethan_one"] == 2)
    # dataset = dataset.remove_columns([ "has_context","bot_morethan_one" , "text" , "metadata", "nb_token" ])

    # dataset_dolphin = load_dataset("ehartford/dolphin",data_files="flan1m-alpaca-uncensored.jsonl" ,split="train")
    # random_test_ids = deterministic_random(42).sample(range(len(dataset_dolphin)), k=10000)
    # dataset_dolphin = dataset_dolphin.select(random_test_ids)
    # dataset_dolphin = dataset_dolphin.map(remove_not_relate_to_thaiEn_dolphin)
    # dataset_dolphin = dataset_dolphin.filter(lambda x: x["has_tran"] == 0)
    # dataset_dolphin = dataset_dolphin.remove_columns(['instruction', 'input', 'output', 'has_tran'])

    dataset_iapp = load_dataset("iapp_wiki_qa_squad", split="train")
    dataset_iapp = dataset_iapp.map(re_formate_iappQA)
    dataset_iapp = dataset_iapp.remove_columns(
        ["question_id", "article_id", "title", "context", "question", "answers"]
    )

    dataset_thaisum = load_dataset("thaisum", split="train")
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_thaisum)), k=20000
    )
    dataset_thaisum = dataset_thaisum.select(random_test_ids)
    dataset_thaisum = dataset_thaisum.map(re_formate_thaisum)
    dataset_thaisum = dataset_thaisum.remove_columns(
        ["title", "body", "summary", "type", "tags", "url"]
    )

    dataset_xlsum = load_dataset("csebuetnlp/xlsum", "thai", split="train")
    # random_test_ids = deterministic_random(42).sample(range(len(dataset_xlsum)), k=5000)
    # dataset_xlsum = dataset_xlsum.select(random_test_ids)
    dataset_xlsum = dataset_xlsum.map(re_formate_xlsum)
    dataset_xlsum = dataset_xlsum.remove_columns(
        ["id", "url", "title", "summary", "text"]
    )

    dataset_enth = load_dataset("scb_mt_enth_2020", "enth", split="train")

    random_test_ids = deterministic_random(42).sample(range(len(dataset_enth)), k=10000)
    dataset_enth = dataset_enth.select(random_test_ids)
    dataset_then = dataset_enth

    dataset_enth = dataset_enth.map(re_formate_scb_enth)
    dataset_enth = dataset_enth.remove_columns(["translation", "subdataset"])

    # dataset_then = load_dataset("scb_mt_enth_2020" , "then",split="train")
    # dataset_then = dataset_then.select(random_test_ids)
    dataset_then = dataset_then.map(re_formate_scb_then)
    dataset_then = dataset_then.remove_columns(["translation", "subdataset"])

    # dataset_alpaca_gpt4 = load_dataset("json", data_files="/workspace/sealion/alpaca_gpt4_data.json")
    # dataset_alpaca_gpt4 = dataset_alpaca_gpt4.rename_column("instruction", "Instruction").rename_column("output", "Answer").rename_column("input", "Context")

    dataset_han = load_dataset(
        "csv",
        data_files="/workspace/sealion/examples/pythainlp-han_dataset - pythainlp-han_dataset.csv",
    )
    dataset_han = dataset_han.map(re_formate_han).remove_columns(["q", "a"])

    all_file = read_file()
    dataset_ = load_dataset("json", data_files=all_file, split="train")
    dataset_1 = dataset_.filter(lambda example: example["split"] == "train")

    dataset_en_th = dataset_1.filter(
        lambda example: example["config"] in ["en_th"]
    )  # ['en_th', 'th', 'thai' ]
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_en_th)), k=20000
    )
    dataset_en_th = dataset_en_th.select(random_test_ids)
    dataset_en_th = dataset_en_th.map(re_formate_xp3)
    dataset_en_th = dataset_en_th.remove_columns(
        ["inputs", "targets", "language", "split", "template", "dataset", "config"]
    )

    dataset_thai = dataset_1.filter(lambda example: example["config"] in ["thai"])
    random_test_ids = deterministic_random(42).sample(range(len(dataset_thai)), k=20000)
    dataset_thai = dataset_thai.select(random_test_ids)
    dataset_thai = dataset_thai.map(re_formate_xp3)
    dataset_thai = dataset_thai.remove_columns(
        ["inputs", "targets", "language", "split", "template", "dataset", "config"]
    )

    dataset_Platypus = load_dataset("garage-bAInd/Open-Platypus", split="train")
    dataset_Platypus = dataset_Platypus.filter(
        lambda example: example["data_source"] != "scienceqa"
        and example["data_source"] != "reclor"
    )  # due to copyright
    dataset_Platypus = dataset_Platypus.map(re_formate_platypus)
    dataset_Platypus = dataset_Platypus.remove_columns(
        ["input", "output", "instruction", "data_source"]
    )

    dataset_tiny = load_dataset("nampdn-ai/tiny-codes", split="train")
    random_test_ids = deterministic_random(42).sample(range(len(dataset_tiny)), k=20000)
    dataset_tiny = dataset_tiny.select(random_test_ids)
    dataset_tiny = dataset_tiny.map(re_formate_code)
    dataset_tiny = dataset_tiny.remove_columns(
        [
            "prompt",
            "main_topic",
            "subtopic",
            "adjective",
            "action_verb",
            "scenario",
            "target_audience",
            "programming_language",
            "common_sense_topic",
            "idx",
            "response",
        ]
    )

    dataset_cot = load_dataset("kaist-ai/CoT-Collection", split="train")
    random_test_ids = deterministic_random(42).sample(range(len(dataset_cot)), k=20000)
    dataset_cot = dataset_cot.select(random_test_ids)
    dataset_cot = dataset_cot.map(re_formate_cot)
    dataset_cot = dataset_cot.remove_columns(
        ["source", "target", "rationale", "task", "type"]
    )

    dataset_dolly = load_dataset(
        "databricks/databricks-dolly-15k",
        split="train",
    )
    dataset_dolly = dataset_dolly.map(re_formate_dolly)
    dataset_dolly = dataset_dolly.remove_columns(
        ["instruction", "context", "response", "category"]
    )

    # print(dataset)
    # print(dataset_dolphin)
    print(dataset_iapp)
    print(dataset_thaisum)
    print(dataset_xlsum)
    print(dataset_enth)
    print(dataset_then)
    print(dataset_en_th)
    print(dataset_thai)
    # print(dataset_alpaca_gpt4["train"])
    print(dataset_han["train"])
    print(dataset_Platypus)
    print(dataset_tiny)
    print(dataset_dolly)
    print(dataset_cot)

    dataset = concatenate_datasets(
        [
            # dataset,
            # dataset_dolphin,
            dataset_iapp,
            dataset_thaisum,
            dataset_xlsum,
            dataset_enth,
            dataset_then,
            dataset_en_th,
            dataset_thai,
            # dataset_alpaca_gpt4["train"],
            dataset_han["train"],
            dataset_Platypus,
            dataset_tiny,
            dataset_dolly,
            dataset_cot,
        ]
    )
    dataset = dataset.map(prepare_dolly_text)
    # print(dataset)

    if split_train_test:
        return split_dataset(args, dataset)
    else:
        return dataset


def load_dolly_V18(
    args, split_train_test=True
):  # V18 more lora512,256, ##user respond, pretrain
    # dataset = load_dataset(
    #        "pythainlp/final_training_set_v1",
    #        split="train",
    #        num_proc=8 if not args.streaming else None,
    #        streaming = args.streaming
    #        )
    dataset = load_from_disk(
        "/project/lt200061-changx/finetune_SFT/loaded_datasets/final_training_set_v1"
    )

    dataset = dataset.map(add_prefix)
    dataset = dataset.filter(lambda x: x["bot_morethan_one"] == 2)
    dataset = dataset.remove_columns(
        ["has_context", "bot_morethan_one", "text", "metadata", "nb_token"]
    )

    # dataset_dolphin = load_dataset("ehartford/dolphin",data_files="flan1m-alpaca-uncensored.jsonl" ,split="train")
    dataset_dolphin = load_from_disk(
        "/project/lt200061-changx/finetune_SFT/loaded_datasets/dolphin"
    )
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_dolphin)), k=10000
    )
    dataset_dolphin = dataset_dolphin.select(random_test_ids)
    dataset_dolphin = dataset_dolphin.map(remove_not_relate_to_thaiEn_dolphin)
    dataset_dolphin = dataset_dolphin.filter(lambda x: x["has_tran"] == 0)
    dataset_dolphin = dataset_dolphin.remove_columns(
        ["instruction", "input", "output", "has_tran"]
    )

    # dataset_iapp = load_dataset("iapp_wiki_qa_squad",split="train")
    dataset_iapp = load_from_disk(
        "/project/lt200061-changx/finetune_SFT/loaded_datasets/iapp_wiki_qa_squad"
    )
    dataset_iapp = dataset_iapp.map(re_formate_iappQA)
    dataset_iapp = dataset_iapp.remove_columns(
        ["question_id", "article_id", "title", "context", "question", "answers"]
    )

    # dataset_thaisum = load_dataset("thaisum",split="train")
    dataset_thaisum = load_from_disk(
        "/project/lt200061-changx/finetune_SFT/loaded_datasets/thaisum"
    )
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_thaisum)), k=5000
    )
    dataset_thaisum = dataset_thaisum.select(random_test_ids)
    dataset_thaisum = dataset_thaisum.map(re_formate_thaisum)
    dataset_thaisum = dataset_thaisum.remove_columns(
        ["title", "body", "summary", "type", "tags", "url"]
    )

    # dataset_xlsum = load_dataset("csebuetnlp/xlsum" , "thai",split="train")
    dataset_xlsum = load_from_disk(
        "/project/lt200061-changx/finetune_SFT/loaded_datasets/xlsum"
    )
    random_test_ids = deterministic_random(42).sample(range(len(dataset_xlsum)), k=5000)
    dataset_xlsum = dataset_xlsum.select(random_test_ids)
    dataset_xlsum = dataset_xlsum.map(re_formate_xlsum)
    dataset_xlsum = dataset_xlsum.remove_columns(
        ["id", "url", "title", "summary", "text"]
    )

    # dataset_enth = load_dataset("scb_mt_enth_2020" , "enth",split="train")
    dataset_enth = load_from_disk(
        "/project/lt200061-changx/finetune_SFT/loaded_datasets/scb_mt_enth_2020"
    )

    random_test_ids = deterministic_random(42).sample(range(len(dataset_enth)), k=10000)
    dataset_enth = dataset_enth.select(random_test_ids)
    dataset_then = dataset_enth

    dataset_enth = dataset_enth.map(re_formate_scb_enth)
    dataset_enth = dataset_enth.remove_columns(["translation", "subdataset"])

    # dataset_then = load_dataset("scb_mt_enth_2020" , "then",split="train")
    # dataset_then = dataset_then.select(random_test_ids)
    dataset_then = dataset_then.map(re_formate_scb_then)
    dataset_then = dataset_then.remove_columns(["translation", "subdataset"])

    # dataset_alpaca_gpt4 = load_dataset("json", data_files="/workspace/sealion/alpaca_gpt4_data.json")
    # dataset_alpaca_gpt4 = dataset_alpaca_gpt4.rename_column("instruction", "Instruction").rename_column("output", "Answer").rename_column("input", "Context")
    # #data_add = add_more_dataset()
    # # print(data_add)
    # print(dataset)
    # print(dataset_dolphin)
    # print(dataset_iapp)
    # print(dataset_thaisum)
    # print(dataset_xlsum)
    # print(dataset_enth)
    # print(dataset_then)
    # print(dataset_alpaca_gpt4)

    dataset_han = load_dataset(
        "csv",
        data_files="/project/lt200061-changx/finetune_SFT/sealion/examples/pythainlp-han_dataset - pythainlp-han_dataset.csv",
    )
    dataset_han = dataset_han.map(re_formate_han).remove_columns(["q", "a"])

    all_file = read_file()
    dataset_ = load_dataset("json", data_files=all_file, split="train")
    dataset_1 = dataset_.filter(lambda example: example["split"] == "train")

    dataset_en_th = dataset_1.filter(
        lambda example: example["config"] in ["en_th"]
    )  # ['en_th', 'th', 'thai' ]
    random_test_ids = deterministic_random(42).sample(
        range(len(dataset_en_th)), k=40000
    )
    dataset_en_th = dataset_en_th.select(random_test_ids)
    dataset_en_th = dataset_en_th.map(re_formate_xp3)
    dataset_en_th = dataset_en_th.remove_columns(
        ["inputs", "targets", "language", "split", "template", "dataset", "config"]
    )

    dataset_thai = dataset_1.filter(lambda example: example["config"] in ["thai"])
    # random_test_ids = deterministic_random(42).sample(range(len(dataset_thai)), k=80000)
    # dataset_thai = dataset_thai.select(random_test_ids)
    dataset_thai = dataset_thai.map(re_formate_xp3)
    dataset_thai = dataset_thai.remove_columns(
        ["inputs", "targets", "language", "split", "template", "dataset", "config"]
    )

    # dataset_Platypus = load_dataset("garage-bAInd/Open-Platypus", split="train")
    dataset_Platypus = load_from_disk(
        "/project/lt200061-changx/finetune_SFT/loaded_datasets/Open-Platypus"
    )
    dataset_Platypus = dataset_Platypus.filter(
        lambda example: example["data_source"] != "scienceqa"
        and example["data_source"] != "reclor"
    )  # due to copyright
    dataset_Platypus = dataset_Platypus.map(re_formate_platypus)
    dataset_Platypus = dataset_Platypus.remove_columns(
        ["input", "output", "instruction", "data_source"]
    )

    # dataset_dolly = load_dataset(
    #         "databricks/databricks-dolly-15k",
    #         split="train",
    #         )
    # dataset_dolly = dataset_dolly.map(re_formate_dolly)
    # dataset_dolly = dataset_dolly.remove_columns(['instruction', 'context', 'response', 'category'])
    dataset_thai_sample_500k = load_from_disk(
        "/project/lt200061-changx/finetune_SFT/loaded_datasets/thai_sample_500k"
    )
    dataset_thai_sample_500k = dataset_thai_sample_500k.map(re_formate_thai_sample_500k)
    dataset_thai_sample_500k = dataset_thai_sample_500k.remove_columns(["text"])

    print(dataset)
    print(dataset_dolphin)
    print(dataset_iapp)
    print(dataset_thaisum)
    print(dataset_xlsum)
    print(dataset_enth)
    print(dataset_then)
    print(dataset_en_th)
    print(dataset_enth)
    print(dataset_thai)
    print(dataset_han["train"])
    print(dataset_Platypus)
    print(dataset_thai_sample_500k)

    dataset = concatenate_datasets(
        [
            dataset,
            dataset_dolphin,
            dataset_iapp,
            dataset_thaisum,
            dataset_xlsum,
            dataset_enth,
            dataset_then
            # , dataset_alpaca_gpt4["train"],
            ,
            dataset_en_th,
            dataset_thai
            # dataset_dolly,
            ,
            dataset_han["train"],
            dataset_Platypus,
            dataset_thai_sample_500k,
        ]
    )

    dataset = dataset.map(prepare_dolly_text)
    # print(dataset)

    if split_train_test:
        return split_dataset(args, dataset)
    else:
        return dataset


def count_tokenizer(tokenizer, dataset):
    llll = 0
    all_token_ids = []
    for tokenized_input111 in dataset:
        tokenized_inputs = tokenizer(
            tokenized_input111["Instruction"], truncation=False
        )["input_ids"]
        tokenized_ans = tokenizer(tokenized_input111["Answer"], truncation=False)[
            "input_ids"
        ]
        llll = llll + len(tokenized_inputs) + len(tokenized_ans)
    return llll


def add_prefix(example):
    a = example["text"].split("<bot>:")

    example["bot_morethan_one"] = len(a)
    example["has_context"] = 1 if "<context>:" in example["text"] else 0

    v = example["text"]

    # Find the indices of the tags
    context_index = v.find("<context>:")
    human_index = v.find("<human>:")
    bot_index = v.find("<bot>:")

    context = v[context_index:human_index].replace("<context>:", "").strip()
    human = v[human_index:bot_index].replace("<human>:", "").strip()
    bot = v[bot_index:].replace("<bot>:", "").strip()

    combined = ""
    if context != "":
        combined = context + "\n" + human
    else:
        combined = human

    example["Context"] = ""
    example["Instruction"] = combined.strip()
    example["Answer"] = bot.strip()

    return example


def prepare_dolly_text(example):
    # if example["Context"] != "":
    #     text = prompt_format_with_context.format(
    #         context=example["Context"],instruction=example["Instruction"], response=example["Answer"]
    #     )

    # else:
    #     text = prompt_format_without_context.format(
    #         instruction=example["Instruction"], response=example["Answer"]
    #     )
    # if example["Answer"] != "":
    #     text = prompt_format_without_context.format(
    #         instruction=example["Instruction"], response=example["Answer"]
    #     )
    # else:
    #     text = example["Instruction"]

    # text = prompt_format_without_context.format(
    #         instruction=example["Instruction"], response=example["Answer"]
    #    )
    if example["Answer"] != "":  # V21
        text = example["Instruction"] + "\n" + example["Answer"]
    else:
        text = example["Instruction"]

    example["text"] = text
    return example


def create_datasets(tokenizer, conf):
    train_data_list = []
    valid_data_list = []

    train_data, valid_data = load_dolly(conf, tokenizer)

    if conf.streaming:
        print("Loading the dataset in streaming mode")
        train_data = train_data.shuffle(buffer_size=conf.shuffle_buffer, seed=None)
    else:
        train_data = train_data.shuffle(seed=None)
        print(
            f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
        )

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        dataset_text_field="text",
        infinite=True,
        seq_length=conf.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        dataset_text_field="text",
        infinite=False,
        seq_length=conf.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset
