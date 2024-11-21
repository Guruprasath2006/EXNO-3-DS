
## EXNO-3-DS
# NAME : Guru prasath K M
# Ref No: 24900987

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:

STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:

1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:

  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```

import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```

![image](https://github.com/SABARI005/EXNO-3-DS/assets/118660461/d0858872-6241-48fb-aa5e-f1423ddca7d2)



```py
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/SABARI005/EXNO-3-DS/assets/118660461/68e025ec-ce14-4c88-b9c7-5cabdad65753)



```py
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/SABARI005/EXNO-3-DS/assets/118660461/c23b9f24-a16f-4a1d-9547-0bbec65480e4)



```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/SABARI005/EXNO-3-DS/assets/118660461/9823a604-315b-4196-96f3-c95f58f2955c)


```py
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```


```py
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/SABARI005/EXNO-3-DS/assets/118660461/d921388b-498a-4ad7-a547-8d8453bd075a)




```py
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/SABARI005/EXNO-3-DS/assets/118660461/9756d37e-f998-470d-aaa2-703cb7a34c00)



```py
pip install --upgrade category_encoders
```

```py
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```


```py
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```


```py
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/SABARI005/EXNO-3-DS/assets/118660461/604795a6-0b00-45cb-a633-fa75b7f96c1d)




```py
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/SABARI005/EXNO-3-DS/assets/118660461/b0f28144-eca5-486c-9cee-0ade91cb7348)



```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/SABARI005/EXNO-3-DS/assets/118660461/fc355faa-b49f-46d1-9221-023c62b5696c)



```py
df.skew()
```
![image](https://github.com/SABARI005/EXNO-3-DS/assets/118660461/ad080da8-dcce-457c-a065-ae4e15eace94)


```py
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/SABARI005/EXNO-3-DS/assets/118660461/48fc9c98-2ffd-456d-9ea9-bfa4d3cac9c0)


```py
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/SABARI005/EXNO-3-DS/assets/118660461/f72c2080-48df-49f2-bd06-5e8fe5dd6096)




```py
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/SABARI005/EXNO-3-DS/assets/118660461/f2740ad1-87fc-4c99-9966-5aaa84e967b2)



```py
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/SABARI005/EXNO-3-DS/assets/118660461/a8f52f95-b7f2-4121-b23b-c972c87f9231)



```py
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/SABARI005/EXNO-3-DS/assets/118660461/be84fd82-b632-408b-a95e-68d5635dce4f)



```py
df.skew()
```
![image](https://github.com/PriyankaAnnadurai/EXNO-3-DS/assets/118351569/09de5543-18fc-4968-8aa9-1082f8610b8d)


```py
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/PriyankaAnnadurai/EXNO-3-DS/assets/118351569/5a2f96dc-6105-4cd7-aa27-69a1893c0cdc)

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/PriyankaAnnadurai/EXNO-3-DS/assets/118351569/092734aa-52a6-4b13-b25f-d9cf2d8eb45d)

```py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/PriyankaAnnadurai/EXNO-3-DS/assets/118351569/45da37db-b7e8-4866-bbcd-ff6b9b429557)


```py
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/PriyankaAnnadurai/EXNO-3-DS/assets/118351569/8d5490fa-4651-47c8-bd6f-d42bc6c8bb1c)



```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/PriyankaAnnadurai/EXNO-3-DS/assets/118351569/9cc1839f-34f1-47e5-a757-bb4540a4cd2f)



```py
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![319878460-6f7a4eaa-1c54-4409-8b57-8b407d5842f7](https://github.com/PriyankaAnnadurai/EXNO-3-DS/assets/118351569/5d511ce4-1999-414c-84a6-d3989e63aa6e)


```py
dt=pd.read_csv("titanic_dataset.csv")
dt
```

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```
![319878630-e2ff6572-cb52-434f-8d9a-980843e1a1b9](https://github.com/PriyankaAnnadurai/EXNO-3-DS/assets/118351569/33e7bf1e-09f0-41cf-9733-bc39e2a82947)

```py
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![319878540-d5c66705-7e21-4a6b-8bc7-a9e1ca23ae85](https://github.com/PriyankaAnnadurai/EXNO-3-DS/assets/118351569/f0d0c362-034d-4938-8b05-9dcddd9c94b6)



# RESULT:

       Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfull

       
