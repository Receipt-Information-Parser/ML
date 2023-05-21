import json
from PIL import Image
import labelbox
import os
import pandas as pd
import ast 

class Preprocess:
  def __init__(self,Data_path):
    self.data_path=Data_path
    # 파일 목록을 받아올 디렉토리 경로를 지정합니다.
    self.js_directory_path = os.path.join(Data_path,"train_data_json")
    self.img_directory_path = os.path.join(Data_path,"train_data_image")


  
  def js_open(self,js_file_path,img):  
    with open(js_file_path) as json_:
      load_json = json.load(json_)

    event=load_json
    images=load_json['images'][0]
    text=[]
    location=[]
    
    for i in event['images'][0]['fields']:
      text.append(str(i['inferText']))
      location.append([int(i['boundingPoly']['vertices'][0]['x']),int(i['boundingPoly']['vertices'][0]['y']),int(i['boundingPoly']['vertices'][1]['x']),int(i['boundingPoly']['vertices'][2]['y'])])

    table_lable=[0 for _ in range(len(text))]

    embedding_loaction=[]
    #print('img.size',img.size)
    for j in range(len(location)):
      temp=[]
      temp.append(int(location[j][0]/img.size[0]*1000))
      temp.append(int(location[j][1]/img.size[1]*1000))
      temp.append(int(location[j][2]/img.size[0]*1000))
      temp.append(int(location[j][3]/img.size[1]*1000))
      for z in range(len(temp)):
        if temp[z]<0:
          temp[z]=0

      embedding_loaction.append(temp)
    #print(embedding_loaction)
    return text, table_lable, embedding_loaction


  def get_label_box_data(self):
    # Enter your Labelbox API key here
    LB_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGU5cDhnb2oxMWR3MDd1MjNjem5jdndoIiwib3JnYW5pemF0aW9uSWQiOiJjbGU5cDhnbzMxMWR2MDd1MjBrbGEweGc2IiwiYXBpS2V5SWQiOiJjbGZhYXNxeDcxbjRkMDd3bDJpaXUwc2ZlIiwic2VjcmV0IjoiNWRhZGZiYTU0YzEwODVkZjU5YWU4ODYwYzY5YzRiYmQiLCJpYXQiOjE2Nzg5MjE5ODAsImV4cCI6MjMxMDA3Mzk4MH0.7qbiis-IRQj-2jkJqBzjlCBH0bnAHdL0WXMfdb0jYfA"
    lb = labelbox.Client(api_key=LB_API_KEY)
    # Get project by ID
    project = lb.get_project('clfuz4j1o091g072x08641fpp')

    # Export labels created in the selected date range as a json file:
    label_box = project.export_labels(download = True, start="2023-05-09", end="2023-05-15")
    return label_box


  def get_json_files(self,directory):
      json_files = []
      for file in os.listdir(directory):
          if file.endswith(".json"):
              json_files.append(file)
      return json_files
    
  def get_img_files(self,directory):
      img_files = []
      for file in os.listdir(directory):
          if file.endswith(".jpeg"):
              img_files.append(file)
      return img_files
    
  def model_preprocessing(self,train_df,label_box):
    box_loc=[ast.literal_eval(i) for i in train_df['bboxes']]
    text=[ast.literal_eval(i) for i in train_df['words']]
    image_path=[i for i in train_df['image_path']]
    tags=[]
    objects={}
    for i in image_path:
      image_name=i
      for j in range(len(label_box)):
        if image_name == label_box[j]['External ID']:
          temp_objects={}
          for h in range(len(label_box[j]['Label']['objects'])):
            temp_objects[label_box[j]['Label']['objects'][h]['value']]=label_box[j]['Label']['objects'][h]['bbox']
          objects[image_name]=temp_objects



    labels = ['others','key','value','total','column_name','item','count','money']
    for i in range(len(box_loc)):
      tags.append([ 0 for _ in range(len(box_loc[i]))])
      
    


    for i in objects.keys():
      image_size=Image.open(os.path.join(self.img_directory_path,str(i))).size
      for j in objects[i].keys():
        objects[i][j]=[objects[i][j]['left']/image_size[0]*1000,objects[i][j]['top']/image_size[1]*1000,(objects[i][j]['left']+objects[i][j]['width'])/image_size[0]*1000,(objects[i][j]['top']+objects[i][j]['height'])/image_size[1]*1000]
    
    for i in range(len(box_loc)):
      box_loc[i]=box_loc[i]


    for i in range(len(image_path)):
      image_name=image_path[i]
      for j in range(len(box_loc[i])):
        x1,x2,y1,y2=box_loc[i][j][0],box_loc[i][j][2],box_loc[i][j][1],box_loc[i][j][3]
        for h in objects[image_name].keys():
          if x1 >= objects[image_name][h][0]-15 and y1 >= objects[image_name][h][1]-15 and x2 <= objects[image_name][h][2]+20 and y2 <= objects[image_name][h][3]+15:
            
            tags[i][j]=labels.index(h)


    train_df['ner_tags']=tags
    train_df.to_csv('/home/guest/ML/dataset/train_data.csv',index=False)
    
  def convert_list_df(self,id_list,text_list,bbox_list,tag_list,image_path_list,remove_index):
    df=pd.DataFrame(id_list,columns=['id'])
    df['words']=text_list
    df['bboxes']=bbox_list
    df['ner_tags']=tag_list
    df['image_path']=image_path_list
    df=df.drop(remove_index,axis=0)
    df.to_csv('/home/guest/ML/dataset/train_data.csv',index=False)
    return print("preprocessing is over")
  
  def do_preprocess(self):
    # get_json_files,get_img_files 함수를 사용하여 파일 목록을 받아옵니다.
    js_file_lists = self.get_json_files(self.js_directory_path)
    img_file_lists=self.get_img_files(self.img_directory_path)
    
    js_file_lists = sorted(js_file_lists, key=lambda x: int(''.join(filter(str.isdigit, x))))
    img_file_lists = sorted(img_file_lists, key=lambda x: int(''.join(filter(str.isdigit, x))))

    total_text=[]
    total_table_lable=[]
    total_embedding_loaction=[]
    for i in range(len(js_file_lists)):
      img=Image.open(os.path.join(self.img_directory_path,img_file_lists[i]))
      output1,output2,output3=self.js_open(os.path.join(self.js_directory_path,js_file_lists[i]),img)
      total_text.append(output1)
      total_table_lable.append(output2)
      total_embedding_loaction.append(output3)
      

    remove_index=[]
    for i in range(len(total_embedding_loaction)):
      for j in range(len(total_embedding_loaction[i])):
        if (total_embedding_loaction[i][j][2]-total_embedding_loaction[i][j][0]) < 0 or (total_embedding_loaction[i][j][3]-total_embedding_loaction[i][j][1])< 0 :
          remove_index.append(i)
      
      

    id_list=[i for i in range(1,len(img_file_lists)+1)]

    self.convert_list_df(id_list,total_text,total_embedding_loaction,total_table_lable,img_file_lists,remove_index)

    train_df=pd.read_csv('/home/guest/ML/dataset/train_data.csv')

    label_box =self.get_label_box_data()

    self.model_preprocessing(train_df,label_box)
    
