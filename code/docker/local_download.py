from transformers import AutoProcessor, AutoModelForTokenClassification

tokenizer = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base")
model = AutoModelForTokenClassification.from_pretrained("seungwon12/cloud_computing_project")


model.save_pretrained('./model')
tokenizer.save_pretrained('./process')