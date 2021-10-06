These demo scripts contain all the modules for the bert-based ner model.
refer to google BERT https://github.com/google-research/bert for more detailed tutorial.
refer to bioBERT https://github.com/dmis-lab/biobert for other options of inital weights.

========================================================
#run ner_pred.py 代码示例：
cd  DIRECTORY_OF_"ner_pred.py"
CUDA_VISIBLE_DEVICES=0  python ner_pred.py  --data_dir=./inputs  --vocab_file=./config/vocab.txt --bert_config_file=./config/bert_config.json  --init_checkpoint=./config/model.ckpt-62717  --output_dir=./outputs

