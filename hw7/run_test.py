import json
from os import path, environ
from argparse import ArgumentParser
from dataset import QADataset
from tqdm import tqdm
from transformers import BertTokenizerFast
from trainer import QATrainer
import pandas as pd
from transformers import QuestionAnsweringPipeline


environ["CUDA_VISIBLE_DEVICES"] = "0"


def predict_qa_by_pipeline(args):
    model = QATrainer.load_from_checkpoint(args.qa_ckpt).to(args.device)
    tokenizer = BertTokenizerFast.from_pretrained(model.hparams['pretrained'])
    model.eval()
    data = json.load(open(args.file))
    context = data['paragraphs']
    questions = data['questions']
    ids = [qid['id'] for qid in questions]
    text = [context[qid['paragraph_id']] for qid in questions]
    question = [qid['question_text'] for qid in questions]
    pipe = QuestionAnsweringPipeline(model.model,
                                     tokenizer,
                                     device=0,
                                     batch_size=32)
    result = pipe(
        question=question,
        context=text,
        max_seq_len=512,
        max_answer_len=20,
    )
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
    args = parser.parse_args()

    result = predict_qa_by_pipeline(args)

    df = pd.DataFrame({
        'ID': list(result.keys()),
        'Answer': list(result.values())
    })
    df.set_index('id', inplace=True)
    df.to_csv(args.output)
