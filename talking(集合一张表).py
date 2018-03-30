import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('gender_age_train.csv',dtype={'device_id': np.str})


targetencoder = LabelEncoder().fit(train.group)
train['group'] = targetencoder.transform(train.group)
nclasses = len(targetencoder.classes_)

test = pd.read_csv('gender_age_test.csv',dtype={'device_id': np.str})
events = pd.read_csv('events.csv',parse_dates=['timestamp'], dtype={'device_id': np.str})

pinpai = pd.read_csv('phone_brand_device_model.csv', dtype={'device_id': np.str})
phone_pinpai = pinpai.drop_duplicates('device_id',keep='first')
phone_pinpai['phone_model'] = phone_pinpai.phone_brand.str.cat(phone_pinpai.device_model)

brandencoder = LabelEncoder().fit(phone_pinpai['phone_brand'])
phone_pinpai['phone_brand'] = brandencoder.transform(phone_pinpai['phone_brand'])

brandencoder = LabelEncoder().fit(phone_pinpai['phone_model'])
phone_pinpai['phone_model'] = brandencoder.transform(phone_pinpai['phone_model'])

app_labels = pd.read_csv('app_labels.csv')
app_label = app_labels.drop_duplicates('app_id',keep='first')

labels = pd.read_csv('label_categories.csv')
labels = labels.fillna('unknown')
brandencoder = LabelEncoder().fit(labels['category'])
labels['category'] = brandencoder.transform(labels['category'])


reader = pd.read_csv('app_events.csv',iterator=True)
loop = True
chunkSize = 1000000
chunks = []
while loop:
    try:
        chunk = reader.get_chunk(chunkSize)
        chunks.append(chunk)
    except StopIteration:
        loop = False
        print ("Iteration is stopped.")
app_events = pd.concat(chunks,ignore_index=True)


#训练数据,品牌,手机型号
df = train.merge(phone_pinpai, how='left',on='device_id')



#app事件和事件聚合
df_shijian = events.merge(app_events, how='left', on='event_id')

#app标签和类别聚合
df_biaoqian = app_label.merge(labels, how='left', on='label_id')

df = df.merge(df_shijian, how='left',on='device_id')
df = df.merge(df_biaoqian, how='left', on='app_id')
#一张表
