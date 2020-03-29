experiment=${1?Error: no experiment name given}
mkdir ../models/$experiment
python -u train.py --max_epochs 10 --valid_iter 1200 --data_path ../../clarification_questions_dataset/data --model_save_path ../models/$experiment/model.pk --vocab_file ../../clarification_questions_dataset/aux/vocab.pkl
python -u test.py --data_path ../../clarification_questions_dataset/data --model_path ../models/$experiment/model.pk --vocab_file ../../clarification_questions_dataset/aux/vocab.pkl
