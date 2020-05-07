experiment=${1?Error: no experiment name given}
mkdir ../trained_models/$experiment
python -u train.py --data_path ../../clarification_questions_dataset/data --model_save_path ../trained_models/$experiment/model.pk --vocab_file ../../clarification_questions_dataset/aux/vocab.pkl
python -u test.py --data_path ../../clarification_questions_dataset/data --model_path ../trained_models/$experiment/model.pk
