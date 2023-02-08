
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned GPT-2 on your custom dataset.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--groundtruth_path', default='afs/numericNLG/data/TD_test_gold', type=str, help='')
    parser.add_argument('--generate_path', default='afs/numericNLG/generated_result/gpt2-en_30_3_1e-6/2/test.out', type=str, help='')
    parser.add_argument('--evaluate_path', default='afs/numericNLG/evaluated_result/gpt2-en_30_3_1e-6/2/test.out', type=str, help='')
    args = parser.parse_args()