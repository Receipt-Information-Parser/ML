try:
    import torch
    import json
    import numpy as np
    from transformers import AutoProcessor, AutoModelForTokenClassification
    from io import BytesIO
    from PIL import Image, ImageDraw, ImageFont
    import os
    import base64
    import requests
    import time
    import io
    import pandas as pd
    import boto3
    import cgi
    import uuid
    import urllib.request
    # Your code goes here
    
except ImportError as e:
    # Handle the import error
    print("Error importing module:", str(e))
except Exception as e:
    # Handle any other exception
    print("An error occurred:", str(e))

class Predict:
    def __init__(self,Model_path,img):
        
        self.label_list= ['others','key','value','total','column_name','item','count','money']
        # 전처리를 위한 모델 호출
        self.processor =  AutoProcessor.from_pretrained("/opt/ml/process", apply_ocr=False)
        self.model=AutoModelForTokenClassification.from_pretrained("/opt/ml/model", num_labels=len(self.label_list))
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

        #print(result)
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
                    if abs(embedding_loaction[index][2]-embedding_loaction[index+1][0]) < 4:
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
        
        device = torch.device('cpu')
        print("here is devide",device)
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
        #print('predictions',len(predictions),predictions)


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
        #print(len(true_predictions),true_predictions)
        for prediction, box in zip(true_predictions, true_boxes):
            
            #if box!=pre and prediction!=-100:
            #  real_output.append(prediction) 
            predicted_label = prediction
            draw.rectangle(box, outline=label2color[predicted_label])
            draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)
            
            pre=box
            
        # self.image.save(os.path.join('/home/guest/ML/code/predict_data/output/','output.jpeg'))
        #true_boxes,true_predictions,text
        return true_boxes,true_predictions,preprocess_data['words']
    def predict2output(self,true_boxes,true_predictions,text):
        #겹치는 예측값 제거 
        real_word_box=[]
        pre=0
        for i in range(len(true_boxes)):
            if sum(true_boxes[i])!=pre:
                real_word_box.append(true_predictions[i])
                pre=sum(true_boxes[i])
            else:
                continue
            
            
            
            
        
        
        combined_value = []
        current_value = ''
        current_x_start = None
        current_x_end = None
        current_y_start = None
        current_y_end = None
        current_layout = None

        for value, layout, position in zip(text, real_word_box, true_boxes):
            if current_layout and layout != current_layout:
                combined_value.append({
                    'value': current_value.strip(),
                    'position': [current_x_start, current_y_start, current_x_end, current_y_end],
                    'layout': current_layout
                })
                current_value = ''
            
            current_value += value + ' '
            current_x_start = position[0] if current_x_start is None else min(current_x_start, position[0])
            current_x_end = position[2] if current_x_end is None else max(current_x_end, position[2])
            current_y_start = position[1] if current_y_start is None else min(current_y_start, position[1])
            current_y_end = position[3] if current_y_end is None else max(current_y_end, position[3])
            current_layout = layout

        combined_value.append({
            'value': current_value.strip(),
            'position': [current_x_start, current_y_start, current_x_end, current_y_end],
            'layout': current_layout
        })
        #print(combined_value)
        
        
        
        value = []
        position = []
        layout = []

        for item in combined_value:
            value.append(item['value'])
            position.append(item['position'])
            layout.append(item['layout'])

        text=value
        real_word_box=layout
        true_boxes=position
        
        
        output_dict={}
        for i in self.label_list:
            output_dict[i]=[]
     

     
        # 예측값에 따라 dict 형태로 예측값 저장

        for i in range(len(real_word_box)):
            if str(real_word_box[i])=='others':
                output_dict['others'].append(text[i])

            elif str(real_word_box[i])=='key':
                output_dict['key'].append(text[i])
            elif str(real_word_box[i])=='value':
                output_dict['value'].append(text[i])
            elif str(real_word_box[i])=='total':
                output_dict['total'].append(text[i])
            elif str(real_word_box[i])=='column_name':
                output_dict['column_name'].append(text[i])
            elif str(real_word_box[i])=='item':
                output_dict['item'].append(text[i])
            elif str(real_word_box[i])=='count':
                output_dict['count'].append(text[i])
            elif str(real_word_box[i])=='money':
                output_dict['money'].append(text[i])

        new_key = []
        previous_word = None


        print("output is : \n",output_dict)
        try:
            key_list=output_dict['key']
            value_list = output_dict['value']
            combine_dict={}
            for i in range(len(key_list)):
                combine_dict[key_list[i]]=value_list[i]
            
            output_dict['key_value']=[combine_dict]
            del output_dict['key']
            del output_dict['value']
            
        except Exception as e:
            print(f"excetion occurred: {str(e)}")
        
        output_dict['csv']=[]
            
        try:
            new_dict = {
                #'column_name': output_dict['column_name'],
                'item': output_dict['item'],
                'count': output_dict['count'],
                'money': output_dict['money']
            }
            #품목 수량 단위 금액 
            max_length = max( len(new_dict['item']), len(new_dict['count']), len(new_dict['money']))
            #len(new_dict['column_name']),
            
            #new_dict['column_name'] += [''] * (max_length - len(new_dict['column_name']))
            print('max_length',max_length)
            for key in ['item', 'count', 'money']:
                new_dict[key] += [''] * (max_length - len(new_dict[key]))
            print('new_dict',new_dict)
            
            for indx in range(len(new_dict['money'])):
                new_dict['money'][indx] = new_dict['money'][indx].replace(',', '')  # Remove commas from the string
                new_dict['money'][indx] = new_dict['money'][indx].replace('.', '')
                new_dict['money'][indx] = new_dict['money'][indx].replace('원', '')  # Remove the '원' symbol from the string
                new_dict['money'][indx]=int(new_dict['money'][indx])
                
            
            #col_name=new_dict['column_name']
            #del new_dict['column_name']
                        
            col_name=['품목','수량','금액']
            
            df=pd.DataFrame.from_dict(new_dict)
            count_list = []
            df.columns=col_name
            import re
            for index, row in df.iterrows():
                match = re.search(r"(\d+)(g|ml|kg|l)", row['품목'])
                if match:
                    quantity = int(match.group(1))
                    unit = match.group(2)
                    print("Item:", row['품목'])
                    print("Quantity:", quantity)
                    print("Unit:", unit)
                    print()
                    count_list.append(quantity)
                    if unit == 'g':
                        df.at[index, '수량'] *= quantity
                    elif unit == 'ml':
                        df.at[index, '수량'] *= quantity
                    elif unit == 'kg':
                        df.at[index, '수량'] *= quantity * 1000
                    elif unit == 'l':
                        df.at[index, '수량'] *= quantity * 1000
                else:
                    count_list.append('개')
            
            df['단위'] = count_list

            # Change the column order
            df = df[['품목', '수량', '단위', '금액']]


                
            
            csv_string = df.to_csv(index=False)
            print('output_dict',output_dict)
            return_dict={}
            if len(output_dict['total']) ==0:
                new_notnull_df=df.dropna()
                print('new_notnull_df',new_notnull_df)
                output_dict['total']=sum(new_notnull_df['금액'])
            
            # Return output_dict and DataFrame as CSV string
            else:
                
            
                return_dict['total']=output_dict['total']
                
                return_dict['total'][0] = return_dict['total'][0].replace(',', '')  # Remove commas from the string
                return_dict['total'][0] = return_dict['total'][0].replace('.', '')
                return_dict['total'][0] = return_dict['total'][0].replace('원', '')
                return_dict['total'][0] = return_dict['total'][0].replace(' ', '')
                return_dict['total'][0]=int(return_dict['total'][0])
                
                
            return_dict['key_value']=output_dict['key_value']
            
            
            
            payment_time=''
            for item in output_dict['others']:
                if '결제일시' in item:
                    payment_time = item
                    break
            
            return_dict['date']=payment_time
            
            
            
            date_string = return_dict['date']
            match = re.search(r"결제일시:\s+(\d{4}-\d{2}-\d{2})", date_string)
            if match:
                date = match.group(1)
                print(date)
                return_dict['date']=date
            else:
                print("No date found.")
                return_dict['date']=''
                
            return_dict['csv']=csv_string
            
            
            return return_dict

        except Exception as e:
            # Handle the exception
            print("There is empty output in the predict data (model accuracy is not good)")
            print(f"An exception occurred: {str(e)}")
            return output_dict

import cgi
def get_file_from_request_body(headers, body):
    fp = io.BytesIO(base64.b64decode(body)) # decode
    environ = {"REQUEST_METHOD": "POST"}
    headers = {
        "content-type": headers["content-type"],
        "content-length": headers["content-length"],
    }

    fs = cgi.FieldStorage(fp=fp, environ=environ, headers=headers) 
    
    print("FS: ", fs)
    #print(fs["file"])
    
    input_file = fs["file"].file.read() # input으로 사용
    model_name = 'seungwon12/cloud_computing_project'
    '''
    model_name = 'seungwon12/cloud_computing_project'
    print(event)
    binary_image = event['body']  # 이미지 이름은 이벤트로부터 받아온다고 가정합니다.
    decode_image=base64.b64decode(binary_image)
    print(decode_image)'''

    img = Image.open(BytesIO(input_file)).convert('RGB')
    predicter = Predict(model_name, img)

    preprocess_data = predicter.predict_preprocess()
    box, predict, text = predicter.predict(preprocess_data)

    output = predicter.predict2output(box, predict, text)
    
    #output_file = json.dumps(output)

    #output_file = output.decode("utf-8")


    return [output, None] 

def lambda_handler(event, context):
    print(event)
    method = event["requestContext"]["http"]["method"]
    # print(method)
    
    if method == "POST":
        
        file_item, file_item_error = get_file_from_request_body(
            headers=event["headers"], body=event["body"]
        )
    else:
        file_item = None
    
    body = json.dumps(file_item, ensure_ascii=False).encode('utf-8')  # Encode the response as UTF-8
    
    body_base64 = base64.b64encode(body).decode('utf-8')  # Convert the UTF-8 encoded response to base64
    
    return {
        'isBase64Encoded': True,
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json; charset=utf-8'  # Specify UTF-8 encoding in the Content-Type header
        },
        'body': body_base64
    }
