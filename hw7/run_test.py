import json
from os import path, environ
from argparse import ArgumentParser
from dataset import QADataset
from tqdm import tqdm
from transformers import BertTokenizerFast, AutoTokenizer
from trainer import QATrainer
import pandas as pd
from transformers import QuestionAnsweringPipeline

from typing import List
from opencc import OpenCC

environ["CUDA_VISIBLE_DEVICES"] = "0"


def to_cn(text: List[str]):
    converter = OpenCC('t2s.json')
    return converter.convert([t for t in text])

def predict_qa_by_pipeline(args):
    model = QATrainer.load_from_checkpoint(args.qa_ckpt).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model.hparams['pretrained'])
    model.eval()
    data = json.load(open(args.file))
    context = data['paragraphs']
    questions = data['questions']
    ids = [qid['id'] for qid in questions]
    text = [context[qid['paragraph_id']] for qid in questions]
    question = [qid['question_text'] for qid in questions]

    if args.cn:
        text = to_cn(text)
        question = to_cn(question)
    pipe = QuestionAnsweringPipeline(model.model,
                                     tokenizer,
                                     device=0,
                                     batch_size=64)
    result = pipe(
        question=question,
        context=text,
        max_seq_len=512,
        max_answer_len=30,
    )
    if args.cn:
        converter = OpenCC('s2t.json')
        dct = {idx: converter.convert(ans['answer']) for idx, ans in zip(ids, result)}
    else:
        dct = {idx: ans['answer'] for idx, ans in zip(ids, result)}

    return dct



if __name__ == '__main__':
    import json
    parser = ArgumentParser()
    parser.add_argument(
        '--qa_ckpt',
        type=str,
        help='QA task ckpt',
        default='adl_hw2/1znn1nmy/checkpoints/epoch=1-step=2713.ckpt',
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
    parser.add_argument('--cn', action='store_true', default=False)
    args = parser.parse_args()
    environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpuid}'

    result = predict_qa_by_pipeline(args)

    df = pd.DataFrame({
        'ID': list(result.keys()),
        'Answer': list(result.values())
    })
    df.set_index('ID', inplace=True)
    df.to_csv(args.output)
