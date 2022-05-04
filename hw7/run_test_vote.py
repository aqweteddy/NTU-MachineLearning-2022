import json
from os import path, environ
from argparse import ArgumentParser
from dataset import QADataset
from tqdm import tqdm
from transformers import BertTokenizerFast, AutoTokenizer
from trainer import QATrainer
import pandas as pd
from transformers import QuestionAnsweringPipeline
from collections import defaultdict
from typing import List
from opencc import OpenCC


def to_cn(text: List[str]):
    converter = OpenCC('t2s.json')
    return [converter.convert(t) for t in text]

def predict_qa_by_pipeline(ckpt, args):
    model = QATrainer.load_from_checkpoint(ckpt).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model.hparams['pretrained'])
    model.eval()
    data = json.load(open(args.file))
    context = data['paragraphs']
    questions = data['questions']
    text = [context[qid['paragraph_id']] for qid in questions]
    question = [qid['question_text'] for qid in questions]

    # if model.hparams['to_cn']:
    #     text = to_cn(text)
    #     question = to_cn(question)
    pipe = QuestionAnsweringPipeline(model.model,
                                     tokenizer,
                                     device=0,
                                     batch_size=64)
    result = pipe(
        question=question,
        context=text,
        max_seq_len=450,
        max_answer_len=30,
        topk=5
    )
    converter = OpenCC('s2t.json')
    # if model.hparams['to_cn']:
    #     for rs in result:
    #         for r in rs:
    #             r['answer'] = converter.convert(r['answer'])
    return result

def ensem(inputs: List[List[dict]]):
    result = []
    for q_model in zip(*inputs): # for each question
        dct = defaultdict(lambda : 0)
        for model_answer in q_model: # for each model
            for i, r in enumerate(model_answer): # for each answer
                # dct[r['answer']] += r['score']
                dct[r['answer']] += 1 / (i + 1)
        result.append(max(dct.items(), key=lambda x: x[1])[0])
        
    return result


def complete_brackets(text: List[str]):
    left_brackets = "\"(【「《"
    right_brackets = "\")】」》"
    for t in text:
        for i, b in enumerate(left_brackets):
            if b in t and right_brackets[i] not in t:
                t += right_brackets[i]
                break
        for i, b in enumerate(right_brackets):
            if b in t and left_brackets[i] not in t:
                t = left_brackets[i] + t
                break
    return text


if __name__ == '__main__':
    import json
    parser = ArgumentParser()
    parser.add_argument(
        '--qa_ckpt',
        nargs='+',
        help='QA task ckpt',
        default=['checkpoints/macbert_cn_ft-mrc.ckpt'],
        # default=['checkpoints/macbert_cn_ft-mrc.ckpt', 'checkpoints/PERT_cn.ckpt'],
    )
    parser.add_argument('--file',
                        type=str,
                        help='data/hw7_test.json',
                        default='data/hw7_test.json')
    parser.add_argument('--output',
                        type=str,
                        help='target csv',
                        default='predict.csv')
    parser.add_argument('--device',
                        type=str,
                        help='cpu or cuda',
                        default='cuda')
    parser.add_argument('--gpuid',
                        type=int,
                        help='cpu or cuda',
                        default=1)
    args = parser.parse_args()
    print(args.qa_ckpt)
    environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpuid}'
    result = []
    for ckpt in tqdm(args.qa_ckpt):
        result.append(predict_qa_by_pipeline(ckpt, args))
    json.dump(result, open('tmp.json', 'w'), ensure_ascii=False, indent=2)
    result = ensem(result)
    result = complete_brackets(result)
    df = pd.DataFrame({
        'ID': list(range(0, len(result))),
        'Answer': result,
    })
    df.set_index('ID', inplace=True)
    df.to_csv(args.output)
