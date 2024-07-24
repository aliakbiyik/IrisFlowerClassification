
# Hikaye:
# Bu veri seti, ünlü Iris veri setidir ve 1936'da Ronald A. Fisher tarafından tanıtılmıştır. 
# Veri seti, üç farklı iris çiçeği türüne (Iris-setosa, Iris-versicolor, ve Iris-virginica) 
# ait sepal uzunluğu, sepal genişliği, petal uzunluğu ve petal genişliği ölçümlerini içermektedir. 
# Her bir iris türünden 50 örnek bulunmaktadır, toplamda 150 gözlem vardır.

# Problem Tanımı:
# Problem: İris Çiçeği Türünün Tahmin Edilmesi
# Amaç, verilen sepal ve petal ölçümleri kullanılarak iris çiçeği türünü tahmin eden bir model geliştirmektir. 
# Bu problem, bir sınıflandırma problemidir ve iris türlerini doğru bir şekilde sınıflandırmayı hedefler.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Veri setini yükleme
iris=pd.read_csv('Iris.csv')

#İlk 5 değeri görüntüleme
print(iris.head())

print(iris.describe())

#Eksik verileri kontrol etme
print(iris.isnull().sum())

#Veriyi standartlaştırma
scaler = StandardScaler()
iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] = scaler.fit_transform(iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
print(iris.head())

#Kategorik verileri sayısal değere dönüştürme
le = LabelEncoder()
iris['Species'] = le.fit_transform(iris['Species'])
print(iris.head())

#Histograms
numeric_columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
iris[numeric_columns].hist(bins=20,figsize=(12,8),layout=(2,2))
plt.suptitle('Histogram of Features')
plt.show()

#Scatter plot
sb.scatterplot(x='SepalLengthCm',y='PetalLengthCm',hue='Species',data=iris,palette='viridis' )
plt.suptitle("Scatter Plot of Sepal Length vs. Petal Length")
plt.show()

#Pair plot
sb.pairplot(iris,hue='Species',palette='viridis')
plt.suptitle("Pair Plot of Iris Dataset",y=1.02)
plt.show()

#Box plot ile aykırı değerleri inceleme
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(2, 2, i)
    sb.boxplot(data=iris, x='Species', y=column)
    plt.title(f"Box Plot of {column} by Species")
plt.tight_layout()
plt.show()

# Özellikler (features) ve etiketler (labels) ayırma
x=iris.drop(columns=['Id','Species']) #Özellikler
y=iris['Species'] #Etiketler

# Veri setini %70 eğitim, %30 test olarak ayırma
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=42)

print("Eğitim seti boyutu:", x_train.shape)
print("Test seti boyutu:", x_test.shape)

# KNN Modelini oluşturma
k=5 #K değerini belirleme(komşu sayısı)
knn=KNeighborsClassifier(n_neighbors=k)
# Modeli eğitim verisi ile eğitme
knn.fit(x_train,y_train)
# Test verileri üzerinde tahminler yapma
y_pred = knn.predict(x_test)

#Modelin performansını değerlendirme
print("KNN Accuracy:", accuracy_score(y_test, y_pred))
print("KNN Classification Report:\n", classification_report(y_test, y_pred))
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
