from datasets import load_metric, load_from_disk,load_dataset,Features, Sequence, ClassLabel, Value, Array2D, Array3D, filesystems, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoProcessor,AutoModelForTokenClassification,AdamW
import ast
from PIL import Image, ImageDraw, ImageFont
from tqdm.notebook import tqdm
class Train:
    def __init__(self):
        self.epoch=1
        self.labels= ['others','key','value','total','column_name','item','count','money']
        self.id2label = {v: k for v, k in enumerate(self.labels)}
        self.label2id = {k: v for v, k in enumerate(self.labels)}
        self.model_name= "microsoft/layoutlmv3-base"
        self.process= AutoProcessor.from_pretrained(self.model_name, apply_ocr=False)
        self.features = Features({
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            'attention_mask': Sequence(Value(dtype='int64')),
            'labels': Sequence(ClassLabel(names=self.labels)),
            'pixel_values': Array3D(dtype="float64", shape=(3, 224, 224)),
        })
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=len(self.labels))
                
        
    def csv_data2datset(self):
        data = load_dataset("csv", data_files='/home/guest/ML/dataset/train_data.csv', split="train")  # path to your file


        # Split into 70% train, 30% test + validation
        train_test_validation = data.train_test_split(test_size=0.2)

        # Split 30% test + validation into half test, half validation
        test_validation = train_test_validation["test"].train_test_split(test_size=0.5)

        # Gather the splits  to have a single DatasetDict

        dataset = DatasetDict(
            {
                "train": data,
                "validation": test_validation["train"],
                "test": data[10],
            }
        )
        dataset['train'].save_to_disk('/home/guest/ML/code/train_data')
        
    def preprocess_data(self,example):
        self.process.feature_extractor.apply_ocr=False
        images=[Image.open("../dataset/train_data_image/"+i).convert('RGB') for i in example['image_path']]
        words = [ast.literal_eval(i) for i in example['words']]
        boxes = [ast.literal_eval(i) for i in example['bboxes']]
        word_labels = [ast.literal_eval(i) for i in example['ner_tags']]
        encoded_inputs = self.process(images, words, boxes=boxes, word_labels=word_labels,
                                    padding="max_length", truncation=True)
        
        
        return encoded_inputs
    
    def train(self):
        self.csv_data2datset()
        dataset = load_from_disk('/home/guest/ML/code/train_data')
        train_dataset = dataset.map(self.preprocess_data, batched=True, remove_columns=dataset.column_names,features=self.features)
        train_dataset.set_format(type = 'torch')
        #모델 훈련 시작
        print('train batch size',len(train_dataset))
        train_dataloader = DataLoader(train_dataset,batch_size=len(train_dataset))
        batch = next(iter(train_dataloader))

        
        optimizer = AdamW(self.model.parameters(), lr= 0.00005) #


        print('train batch size:', len(train_dataset))
        train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), pin_memory=True)
        
        global_step = 0
        num_train_epochs = self.epoch

        self.model.train() 
        for epoch in range(num_train_epochs):  
            print("Epoch:", epoch)
            for batch in tqdm(train_dataloader):
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(**batch) 
                loss = outputs.loss
                print(f"Loss after {global_step} steps: {loss.item()}")

                loss.backward()
                optimizer.step()
                global_step += 1  
        self.model.config.id2label=self.id2label
        self.model.config.label2id=self.label2id
        self.model.save_pretrained("layoutlmv3")

        
        
        
        