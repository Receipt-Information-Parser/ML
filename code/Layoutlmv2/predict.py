from transformers import AutoProcessor,AutoModelForTokenClassification,AdamW
import torch
from PIL import Image, ImageDraw, ImageFont
import os
import base64
import uuid
import requests
import time
import json
import io


class Predict:
    def __init__(self,Model_path,img):
        
        self.label_list= ['others','key','value','total','column_name','item','count','money']
        # 전처리를 위한 모델 호출
        self.processor =  AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
        self.model=AutoModelForTokenClassification.from_pretrained(Model_path, num_labels=len(self.label_list))
        self.image=img
        self.api_url = 'https://yartf43wok.apigw.ntruss.com/custom/v1/18186/0c8ba49a3121aa041c2fa2d2ef06b11b5f61d50a1da7972cae050f982d02cee2/general'
        self.secret_key = 'dVNYclZYY3d3dlFNbFp3UEJFeWJRekZPSElmdE1BQ1c='
        

    def predict_preprocess(self):
        bytes_stream = io.BytesIO()

        # Convert the PIL Image to bytes and store it in the BytesIO stream
        self.image.save(bytes_stream, format='JPEG')  # Specify the desired format (JPEG, PNG, etc.)

        # Read the contents of the BytesIO object
        img = bytes_stream.getvalue()

        file_data=base64.encodebytes(img).decode('utf-8')
        # naver api 를 호출하기 위해서 파일 형식을 json형식으로 api를 호출하며 데이터를 넘긴다.
        request_json = {
            'images': [
                {
                    'format': 'jpg',
                    'name': 'demo',
                    'data': file_data
                }
            ],
            'requestId': str(uuid.uuid4()),
            'version': 'V2',
            'timestamp': int(round(time.time() * 1000)),
            'enableTableDetection': True
        }

        payload = json.dumps(request_json).encode('UTF-8')
        headers = {
        'X-OCR-SECRET': self.secret_key,
        'Content-Type': 'application/json'
        }


    #requests함수를 이용하여 api를 호출 결과값을 result에 저장
        response=requests.post(self.api_url, headers=headers, data = payload)
        result=response.json()

        print(result)
        # naver ocr api 의 반환값으로 이미지 데이터에서 테이블 형식의 데이터를 저장해준다.
        images=result['images'][0]
        result_data_text=[]
        table_lable=[]
        if 'tables' in images.keys():
            for h in range(0,len(images['tables'])):
                table_text=[]
                table_index=[]
                for i in range(0,len(images['tables'][h]['cells'])):
                    
                    table_index.append([images['tables'][h]['cells'][i]['rowIndex'],images['tables'][h]['cells'][i]['columnIndex'],images['tables'][h]['cells'][i]['rowSpan'],images['tables'][h]['cells'][i]['columnSpan'] ])
                    table_raw_span=images['tables'][h]['cells'][i]['rowSpan'] #텍스트 부분에서 텍스트가 길이가 얼마나인지
                    
                    table_col_span=images['tables'][h]['cells'][i]['columnSpan']
                    temp_table_text=[]

                    if len(images['tables'][h]['cells'][i]['cellTextLines']) !=0:
                        for k in range(len(images['tables'][h]['cells'][i]['cellTextLines'])):
                            for j in range(len(images['tables'][h]['cells'][i]['cellTextLines'][k]['cellWords'])):
                                temp_table_text.append(images['tables'][h]['cells'][i]['cellTextLines'][k]['cellWords'][j]['inferText'])


                    table_text.append(temp_table_text)
                    for i in temp_table_text:
                        result_data_text.append(i)
                        table_lable.append(h)


        # naver ocr api 반환값으로 이미지에서 모든 텍스트를 'text' 배열에 저장
        # 그리고 그 텍스트(바운딩 박스)의 위치 정보는 'location' 배열에 저장
        text=[]
        location=[]
        for i in result['images'][0]['fields']:
            text.append(i['inferText'])
            location.append([int(i['boundingPoly']['vertices'][0]['x']),int(i['boundingPoly']['vertices'][0]['y']),int(i['boundingPoly']['vertices'][1]['x']),int(i['boundingPoly']['vertices'][2]['y'])])


            table_lable=[0 for _ in range(len(text))]

        embedding_loaction=[]

        image=self.image
    

        for i in range(len(location)):
            temp=[]
            temp.append(int(location[i][0]/image.size[0]*1000))
            temp.append(int(location[i][1]/image.size[1]*1000))
            temp.append(int(location[i][2]/image.size[0]*1000))
            temp.append(int(location[i][3]/image.size[1]*1000))
            embedding_loaction.append(temp)



        new_embedding_loaction=[]
        for _ in range(3):
            for index , value in enumerate(embedding_loaction):
                #print(text[index],'\t',table_lable[index],'\t',embedding_loaction[index])
                if len(embedding_loaction) <= index +1:
                    break
                else:
                    if abs(embedding_loaction[index][2]-embedding_loaction[index+1][0]) < 10:
                        embedding_loaction[index+1]=[embedding_loaction[index][0],embedding_loaction[index][1],
                                                embedding_loaction[index+1][2],embedding_loaction[index+1][3]]
                        del embedding_loaction[index]

                        text[index+1]=text[index]+' '+text[index+1]
                        del text[index]

                        del table_lable[index+1]

        temp={}
        temp['id'] =0
        temp['words'] = text#text
        temp['bboxes'] = embedding_loaction # make sure to normalize your bounding boxes
        temp['ner_tags']=table_lable
        return temp

    def predict(self,preprocess_data):
        
        encoded_inputs = self.processor(self.image, preprocess_data['words'], boxes=preprocess_data['bboxes'], word_labels=preprocess_data['ner_tags'],
                           padding="max_length", truncation=True, return_tensors="pt")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        labels = encoded_inputs.pop('labels').squeeze().tolist()
        id2label = {v: k for v, k in enumerate(self.label_list)}
        label2id = {k: v for v, k in enumerate(self.label_list)}
        for k,v in encoded_inputs.items():
            encoded_inputs[k] = v.to(device)

        # 예측 진행

        self.model.to(device)
        outputs = self.model(**encoded_inputs)


        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        token_boxes = encoded_inputs.bbox.squeeze().tolist()
        width, height = self.image.size
        print('predictions',len(predictions),predictions)


        def unnormalize_box(bbox, width, height):
            return [
                int(width * (bbox[0] / 1000)),
                int(height * (bbox[1] / 1000)),
                int(width * (bbox[2] / 1000)),
                int(height * (bbox[3] / 1000)),
            ]

        # true_predictions 예측값이 어떤 layout인지를 string 형식으로 반환하여 저장
        # true_boxes 텍스트 정보가 이전에 전처리 과정에서 1000,1000로 축소된것을 다시 원래 위치로 조정

        true_predictions = [id2label[prediction] for prediction, label in zip(predictions, labels) if label != -100]
        true_boxes = [unnormalize_box(box, width, height) for box, label in zip(token_boxes, labels) if label != -100]

        

        draw = ImageDraw.Draw(self.image)

        font = ImageFont.load_default()



        # 각 layout에 해당하는 색 지정
        label2color = {'others':'blue', 'key':'green', 'value':'orange', 'total':'red','column_name':'black','item':'brown','count':'gray','money':'yellow'}



        #이미지 위에 각 텍스트의 바운딩박스 위치별로 layout을 지정하고 색을 지정하여 결과값을 반환
        pre=0
        real_output=[]
        print(len(true_predictions),true_predictions)
        for prediction, box in zip(true_predictions, true_boxes):
            
            #if box!=pre and prediction!=-100:
            #  real_output.append(prediction) 
            predicted_label = prediction
            draw.rectangle(box, outline=label2color[predicted_label])
            draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)
            
            pre=box
        self.image.save(os.path.join('/home/guest/ML/code/predict_data/output/','output.jpeg'))
                