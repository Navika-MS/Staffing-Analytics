# Librar
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.cluster import AgglomerativeClustering
st.set_option('deprecation.showPyplotGlobalUse', False)

#KNN
import pandas as pd
df=pd.read_csv("HRProjectDataset.csv")
df=df.dropna()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['EmploymentStatus']=le.fit_transform(df['EmploymentStatus'])
df['EmploymentStatus']=pd.DataFrame(df['EmploymentStatus'])
df['CareerSwitcher']=le.fit_transform(df['CareerSwitcher'])
df['CareerSwitcher']=pd.DataFrame(df['CareerSwitcher'])
x=df[['Skill Big Data','Skill Degree','Skill Enterprise Tools','Skill Python','Skill SQL','Skill R']]
y=df[['Current Job Title']]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)
from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(n_neighbors=5)
dk=kn.fit(x_train,y_train)
ypred=dk.predict(x_test)
from sklearn.metrics import accuracy_score
ascore=accuracy_score(ypred, y_test)

st.title("Staffing Analytics")
from PIL import Image
img = Image.open("picture.png")
 # display image using streamlit
# width is used to set the width of an image
st.image(img, width=500)

if st.sidebar.button("Dataset"):
    st.write(df)
if st.sidebar.button("Visualization of KNN"):
    st.title("KNN")
    cm=confusion_matrix(ypred,y_test)
    sns.heatmap(cm,annot=True)
    plt.xlabel('predicted')
    plt.ylabel('actual')
    st.pyplot()
    st.write("Text with write")
    st.text("Welcome To GeeksForGeeks!!!")
    st.text('The accuracy score is ')
    st.text(ascore)
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=3)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)    
if st.sidebar.button("Visualization of DECISION TREE"):
    st.title("Decision Tree")
    from sklearn import tree
    import matplotlib.pyplot as plt
    tree.plot_tree(classifier, filled=True, rounded=True)
    st.pyplot()
    cm=confusion_matrix(y_pred,y_test)
    sns.heatmap(cm,annot=True)
    ascore=accuracy_score(y_test, y_pred)
    cm=confusion_matrix(y_pred,y_test)
    plt.xlabel('predicted')
    plt.ylabel('actual')
    st.pyplot()
    classif=classification_report(y_test, y_pred)
    st.write(classif)
    st.text('The accuracy score is ')
    st.text(ascore)    
df1= pd.read_csv("HRProjectDatasetnew.csv")
df1.head()
df1=df1.dropna()
if st.sidebar.button("Visualization of CLUSTERING"):
    st.title("Clustering")
    hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage ='average')
    y_hc = hc.fit_predict(df1)
    df1['cluster'] = pd.DataFrame(y_hc)
    import seaborn as sns
    plt.figure(figsize=(20, 10))
    sns.heatmap(df1.corr(),annot=True)
    st.pyplot()
    X = df1.iloc[:, [3,4]].values
    plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c='red', label ='Cluster 1')
    plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='blue', label ='Cluster 2')
    plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='green', label ='Cluster 3')
    plt.title('Clusters of Customers (Hierarchical Clustering Model)')
    plt.xlabel('Annual Income(k$)')
    plt.ylabel('Spending Score(1-100)')   
    st.pyplot()
if st.sidebar.button("PLOTS"):
    st.title("BAR PLOT")    
    train = pd.read_csv("HRProjectDataset.csv")
    #train.head()
    test = pd.read_csv("HRProjectDataset.csv")
    #train['EmploymentStatus'].value_counts()
    train['Skill Python'].value_counts().plot.bar()
    train['EmploymentStatus'].value_counts(normalize=True).plot.bar(title='EmploymentStatus')
    st.pyplot()
    st.title("PIE PLOT") 
    
    #Aus_Players = 'Smith', 'Finch', 'Warner', 'Lumberchane'    
   # Runs = [42, 32, 18, 24]    
#    explode = (0.1, 0, 0, 0)

    #fig1, ax1 = plt.subplots()    
   # ax1.pie(EmploymentStatus, explode=explode, labels=Aus_Players, autopct='%1.1f%%',    
     #   shadow=True, startangle=90)    
    #ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.    
    
   # plt.show() 
# import the streamlit library
    
if st.sidebar.button("PLOT"):
    # give a title to our app
    st.title('ELIGIBILITY TEST')
    # TAKE WEIGHT INPUT in kgs
    pyth=st.selectbox("Enter your skill in Python",['',1,2,3,4])
    r=st.selectbox("Enter your skill in R",['',1,2,3,4])
    c = st.selectbox("Enter your skill in c",['',1,2,3,4])
    sql = st.selectbox("Enter your skill in SQL",['',1,2,3,4])
    if st.button("Calculate Eligibility"):
        bmi=pyth+r+c+sql   
        st.text("Your Eligibility Score is ")
        st.text(bmi)
        # give the interpretation of BMI index
        if(bmi < 3):
            st.error("You are Not Eligible")
        elif(bmi >= 3 and bmi <= 5):
            st.warning("Your are not Eligible  for this position.Better luck next time ")    
        elif(bmi >= 6):
            st.error("Congratulations!Your are Eligible!")