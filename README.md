# toonsquare_ai_kobert

update new model
* python train_mlm_azure.py --pretrained_type="etri"
* python train_classification_a.py  --pretrained_model_path="./mlm_model/mlm_best_model.bin"

download new model && test
* python app_azure_test.py
