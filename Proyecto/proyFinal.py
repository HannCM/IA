#Biblioteca streamlit 
import streamlit as st #Biblioteca streamlit
from PIL import Image #Biblioteca para manejar imágenes
#Biliotecas generales
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
import seaborn as sns #Visualización de datos

#Menú
menu = ["Inicio", "Apriori", "Métricas de Distancia", "Clustering Particional", "Clustering Jerárquico", "Clasificación", "Árbol de Decisión-Pronóstico","Árbol de Decisión-Clasificación"]
st.sidebar.title("Menú")
option = st.sidebar.selectbox("Selecciona una opción", menu)

#Bienvenida
if option == "Inicio":
    st.title("CERES")
    st.info("¡Qué tal!")
    st.info("En esta aplicación encontrarás algoritmos de aprendizaje automático con los que podrás analizar los datos que desees.")
    image = Image.open('inicio.jpg')
    st.image(image, width=700)
    st.info("¡Pruébala!")

#Apriori
if option == "Apriori":
    st.title("Apriori")
    #Carga de datos
    datosArchivo= st.file_uploader("Selecciona un archivo", type=["csv"])
    st.warning("Por favor sube un archivo .CSV")
    if datosArchivo is not None:
        leerDatos=pd.read_csv(datosArchivo, header=None)
        st.info("Visualización de tus datos")
        st.dataframe(leerDatos)

        #Gráfica frecuencia de los datos
        transacciones=leerDatos.values.reshape(-1).tolist()
        lista=pd.DataFrame(transacciones)
        lista['Frecuencia']=0
        lista = lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
        lista['Porcentaje'] = (lista['Frecuencia'] / lista['Frecuencia'].sum()) #Porcentaje
        lista = lista.rename(columns={0 : 'Item'})
        st.info("Tabla de Frecuencias")
        st.dataframe(lista)
        st.info("Gráfica de Frecuencias")
        st.set_option('deprecation.showPyplotGlobalUse', False) #Para que no muestre el warning
        plt.figure(figsize=(16,20),dpi=300)
        plt.ylabel('Item')
        plt.xlabel('Frecuencia')
        plt.barh(lista['Item'], width=lista['Frecuencia'], color='blue')
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False) #Para que no muestre el warning

        #Aplicación de apriori
        from apyori import apriori #Biblioteca de apriori
        listaA=leerDatos.stack().groupby(level=0).apply(list).tolist()
        st.write("Ingresa el valor para tus reglas: ")
        #Ingreso de reglas
        try:
            min_supp=st.number_input("Soporte:")
            min_conf=st.number_input("Confianza:")
            min_lif=st.number_input("Lift:")
            reglas=apriori(listaA, min_support=min_supp, min_confidence=min_conf, min_lift=min_lif)
            resultadoR=list(reglas)
            st.info("Número de reglas encontradas:")
            st.success(len(resultadoR))
            st.info("Reglas")
            for i in resultadoR:
                #Regla
                Emparejar = i[0]
                items = [x for x in Emparejar]
                st.warning("Regla: " + str(i[0]))
                #Valores resultantes
                st.success("Soporte: " + str(i[1]))
                st.success("Confianza: " + str(i[2][0][2]))
                st.success("Lift: " + str(i[2][0][3]))
        except:
            st.warning("Ingresa un soporte mayor a 0") 

#Métricas de distancia
if option == "Métricas de Distancia":
    st.title("Métricas de Distancia")
    from scipy.spatial.distance import cdist #Cálculo de distancias
    #Carga de datos
    datosArchivo= st.file_uploader("Selecciona un archivo", type=["csv"])
    st.warning("Por favor sube un archivo .CSV")
    if datosArchivo is not None:
        leerDatos=pd.read_csv(datosArchivo)
        st.info("Visualización de tus datos")
        st.dataframe(leerDatos)
        #Elección de métrica
        st.warning("Selecciona una métrica:")
        metrica=st.selectbox("", ["Euclideana", "Manhattan", "Minkowski", "Chebyshev"])
        if metrica == "Euclideana":
            st.info("Métrica: Euclideana")
            #Distancia euclideana
            dEuclideana=cdist(leerDatos, leerDatos, metric='euclidean')
            mEcucideana=pd.DataFrame(dEuclideana)
            st.write("¿Deseas medir un par de registros de tus datos?")
            if st.checkbox("Sí"):
                try:
                    st.write("Selecciona los registros:")
                    from scipy.spatial import distance
                    dato1=st.number_input("Primer registro:")
                    dato2=st.number_input("Segundo registro:")
                    ob1=leerDatos.iloc[int(dato1)]
                    ob2=leerDatos.iloc[int(dato2)]
                    st.info("Distancia euclideana entre los registros")
                    dEuclideana2=distance.euclidean(ob1,ob2)
                    st.success(dEuclideana2)
                except:
                    st.warning("Ingresa registros existentes")
            if st.checkbox("No"):
                st.info("Matriz de Distancias")
                st.dataframe(mEcucideana)
        if metrica == "Manhattan":
            st.info("Métrica: Manhattan")
            #Distancia Manhattan
            dManhattan=cdist(leerDatos, leerDatos, metric='cityblock')
            mManhattan=pd.DataFrame(dManhattan)
            st.write("¿Deseas medir un par de registros de tus datos?")
            if st.checkbox("Sí"):
                try:
                    st.write("Selecciona los registros:")
                    from scipy.spatial import distance
                    dato1=st.number_input("Primer registro:")
                    dato2=st.number_input("Segundo registro:")
                    ob1=leerDatos.iloc[int(dato1)]
                    ob2=leerDatos.iloc[int(dato2)]
                    st.info("Distancia Manhattan entre los registros")
                    dManhattan2=distance.cityblock(ob1,ob2)
                    st.success(dManhattan2)
                except:
                    st.warning("Ingresa registros existentes")
            if st.checkbox("No"):
                st.info("Matriz de Distancias")
                st.dataframe(mManhattan)
        if metrica == "Minkowski":
            st.info("Métrica: Minkowski")
            #Distancia Minkowski
            st.write("Ingresa el valor de la distancia:")
            dMinkowski=cdist(leerDatos, leerDatos, metric='minkowski', p=1.5)
            mMinkowski=pd.DataFrame(dMinkowski)
            st.write("¿Deseas medir un par de registros de tus datos?")
            if st.checkbox("Sí"):
                try:
                    st.write("Selecciona los registros:")
                    from scipy.spatial import distance
                    dato1=st.number_input("Primer registro:")
                    dato2=st.number_input("Segundo registro:")
                    ob1=leerDatos.iloc[int(dato1)]
                    ob2=leerDatos.iloc[int(dato2)]
                    st.info("Distancia Minkowski entre los registros")
                    dMinkowski2=distance.minkowski(ob1,ob2,p=1.5)
                    st.success(dMinkowski2)
                except:
                    st.warning("Ingresa registros existentes")
            if st.checkbox("No"):
                st.info("Matriz de Distancias")
                st.dataframe(mMinkowski)
        if metrica == "Chebyshev":
            st.info("Métrica: Chebyshev")
            #Distancia Chebyshev
            dChebyshev=cdist(leerDatos, leerDatos, metric='chebyshev')
            mChebyshev=pd.DataFrame(dChebyshev)
            st.write("¿Deseas medir un par de registros de tus datos?")
            if st.checkbox("Sí"):
                try:
                    st.write("Selecciona los registros:")
                    from scipy.spatial import distance
                    dato1=st.number_input("Primer registro:")
                    dato2=st.number_input("Segundo registro:")
                    ob1=leerDatos.iloc[int(dato1)]
                    ob2=leerDatos.iloc[int(dato2)]
                    st.info("Distancia Chebyshev entre los registros")
                    dChebyshev2=distance.chebyshev(ob1,ob2)
                    st.success(dChebyshev2)
                except:
                    st.warning("Ingresa registros existentes")
            if st.checkbox("No"):
                st.info("Matriz de Distancias")
                st.dataframe(mChebyshev)
                
#Clustering Particional
if option == "Clustering Jerárquico":
    st.title("Clustering Jerárquico")
    import seaborn as sns #Visualización de datos
    #Carga de datos
    datosArchivo= st.file_uploader("Selecciona un archivo", type=["csv"])
    st.warning("Por favor sube un archivo .CSV")
    if datosArchivo is not None:
        leerDatos=pd.read_csv(datosArchivo)
        st.info("Visualización de tus datos")
        st.dataframe(leerDatos)
        #Elegir la variable a trabajar 
        st.warning("Selecciona la variable para trabajar")
        variable=st.selectbox("", leerDatos.columns)
        #Número de registros con el mismo valor de la variable
        st.write("Número de registros por grupo:")
        st.dataframe(leerDatos.groupby(variable).size())
        #Selección de características
        st.info("Selección de Características")
        st.write("Elige la forma en que quieres realizar tu selección de carácterísticas:")
        if st.checkbox("Gráfico de Dispersión"):
            st.write("Elige 2 variables")
            variable1=st.selectbox("Variable 1:", leerDatos.columns)
            variable2=st.selectbox("Variable 2:", leerDatos.columns)
            sns.scatterplot(x=variable1, y=variable2, data=leerDatos, hue=variable)
            st.info("Gráfico de Dispersión de las variables:")
            plt.xlabel(variable1)
            plt.ylabel(variable2)
            st.pyplot()
        if st.checkbox("Matriz de Correlaciones"):
            st.info("Matriz de correlaciones de tus datos:")
            mCorre=leerDatos.corr(method='pearson')
            st.dataframe(mCorre)
        if st.checkbox("Mapa de Calor"):
            st.info("Mapa de calor de tus datos:")
            plt.figure(figsize=(14,7))
            mCorre=leerDatos.corr(method='pearson')
            mCalor=np.triu(mCorre)
            sns.heatmap(mCorre, cmap="RdBu_r", annot=True, mask=mCalor)
            st.pyplot()
        try:
            st.write("Selecciona las características que deseas incluir en tu modelo:")
            caracteristicas=st.multiselect("", leerDatos.columns)
            st.write("Tus variables elegidas fueron:")
            st.write(caracteristicas)
            #Estandarización de datos
            mVariables= np.array(leerDatos[caracteristicas])
            from sklearn.preprocessing import StandardScaler, MinMaxScaler #Bibliotecas para escalado de datos
            estandarizar=StandardScaler()
            mEstandarizada=estandarizar.fit_transform(mVariables)
            #Algoritmo: Ascendente jerárquico
            st.info("Obtención de Clústeres")
            st.write("Árbol:")
            import scipy.cluster.hierarchy as shc
            from sklearn.cluster import AgglomerativeClustering
            plt.figure(figsize=(10,7))
            plt.xlabel("Observaciones")
            plt.ylabel("Distancia")
            Arbol = shc.dendrogram(shc.linkage(mEstandarizada,method="complete",metric="euclidean"))
            st.pyplot()
            st.write("Por favor ingrese el número de grupos según el árbol anterior")
            try:
                nGrupos=st.number_input("Grupos")
                mJerarquico=AgglomerativeClustering(n_clusters=int(nGrupos), affinity="euclidean", linkage="complete")
                mJerarquico.fit(mEstandarizada)
                leerDatos["ClusterH"]=mJerarquico.labels_
                st.write("Datos etiquetados")
                st.dataframe(leerDatos)
                st.write("Número de datos por Clúster:")
                datosEti=leerDatos.groupby(["ClusterH"])["ClusterH"].count()
                st.dataframe(datosEti)
                #Obtención de centroides
                st.info("Información de cada uno de tus clústeres")
                centroidesJ=leerDatos.groupby(["ClusterH"])[caracteristicas].mean()
                st.dataframe(centroidesJ)
            except:
                st.warning("Ingresa un número de grupos válido")
        except:
            st.warning("Ingresa tus variables para trabajar")
if option == "Clustering Particional":
    st.title("Clustering Particional")
    import seaborn as sns #Visualización de datos
    #Carga de datos
    datosArchivo= st.file_uploader("Selecciona un archivo", type=["csv"])
    st.warning("Por favor sube un archivo .CSV")
    if datosArchivo is not None:
        leerDatos=pd.read_csv(datosArchivo)
        st.info("Visualización de tus datos")
        st.dataframe(leerDatos)
        #Elegir la variable a trabajar 
        st.warning("Selecciona la variable para trabajar:")
        variable=st.selectbox("", leerDatos.columns)
        #Número de registros con el mismo valor de la variable
        st.write("Número de registros por grupo:")
        st.dataframe(leerDatos.groupby(variable).size())
        #Selección de características
        st.info("Selección de características")
        st.write("Elige la forma en que quieres realizar tu selección de carácterísticas:")
        if st.checkbox("Gráfico de Dispersión"):
            st.write("Elige 2 variables")
            variable1=st.selectbox("Variable 1:", leerDatos.columns)
            variable2=st.selectbox("Variable 2:", leerDatos.columns)
            sns.scatterplot(x=variable1, y=variable2, data=leerDatos, hue=variable)
            st.info("Gráfico de dispersión de las variables:")
            plt.xlabel(variable1)
            plt.ylabel(variable2)
            st.pyplot()
        if st.checkbox("Matriz de Correlaciones"):
            st.info("Matriz de correlaciones de tus datos:")
            mCorre=leerDatos.corr(method='pearson')
            st.dataframe(mCorre)
        if st.checkbox("Mapa de Calor"):
            st.info("Mapa de calor de tus datos:")
            plt.figure(figsize=(14,7))
            mCorre=leerDatos.corr(method='pearson')
            mCalor=np.triu(mCorre)
            sns.heatmap(mCorre, cmap="RdBu_r", annot=True, mask=mCalor)
            st.pyplot()
        try:
            st.write("Selecciona las características que deseas incluir en tu modelo:")
            caracteristicas=st.multiselect("", leerDatos.columns)
            st.write("Tus variables elegidas fueron:")
            st.write(caracteristicas)
            #Estandarización de datos
            mVariables= np.array(leerDatos[caracteristicas])
            from sklearn.preprocessing import StandardScaler, MinMaxScaler #Bibliotecas para escalado de datos
            estandarizar=StandardScaler()
            mEstandarizada=estandarizar.fit_transform(mVariables)
            #Algoritmo: Clustering Particional
            st.info("Obtención de Clústeres")
            #Elbow method
            from sklearn.cluster import KMeans #Biblioteca para K-Means
            from sklearn.metrics import pairwise_distances_argmin_min 
            #Definición de k clusters para K-means
            #Se utiliza random_state para inicializar el generador interno de números aleatorios
            SSE = []
            for i in range(2, 12):
                km = KMeans(n_clusters=i, random_state=0)
                km.fit(mEstandarizada)
                SSE.append(km.inertia_)
            #Se grafica SSE en función de k
            plt.figure(figsize=(10, 7))
            plt.plot(range(2, 12), SSE, marker='o')
            plt.xlabel('Cantidad de clusters *k*')
            plt.ylabel('SSE')
            plt.title('Elbow Method')
            st.pyplot()
            #Localización del clúster
            from kneed import KneeLocator
            kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
            st.write("El número de clústeres son:")
            numCluster=kl.elbow
            st.success(numCluster)
            st.write("Gráfica")
            plt.style.use('ggplot')
            st.pyplot(kl.plot_knee())
            mParticional=KMeans(n_clusters=numCluster, random_state=0).fit(mEstandarizada)
            mParticional.predict(mEstandarizada)
            leerDatos["ClusterP"]=mParticional.labels_
            st.write("Datos etiquetados")
            st.dataframe(leerDatos)
            st.write("Número de datos por Clúster:")
            datosEti=leerDatos.groupby(["ClusterP"])["ClusterP"].count()
            st.dataframe(datosEti)
            #Obtención de centroides
            st.info("Información de cada uno de tus clústeres")
            centroidesP=leerDatos.groupby(["ClusterP"])[caracteristicas].mean()
            st.dataframe(centroidesP)
        except:
            st.warning("Ingresa tus variables para trabajar")
if option == "Clasificación":
    st.title("Clasificación (R. Logística)")
    from sklearn import linear_model
    from sklearn import model_selection
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    #Carga de datos
    datosArchivo= st.file_uploader("Selecciona un archivo", type=["csv"])
    st.warning("Por favor sube un archivo .CSV")
    if datosArchivo is not None:
        leerDatos=pd.read_csv(datosArchivo)
        st.info("Visualización de tus datos")
        st.dataframe(leerDatos)
        #Elegir la variable a trabajar 
        st.warning("Selecciona la variable para trabajar:")
        variable=st.selectbox("", leerDatos.columns)
        #Número de registros con el mismo valor de la variable
        st.write("Número de registros por grupo:")
        st.dataframe(leerDatos.groupby(variable).size())
        #Selección de características
        st.info("Selección de Características")
        st.write("Elige la forma en que quieres realizar tu selección de carácterísticas:")
        if st.checkbox("Gráfico de Dispersión"):
            st.write("Elige 2 variables")
            variable1=st.selectbox("Variable 1:", leerDatos.columns)
            variable2=st.selectbox("Variable 2:", leerDatos.columns)
            sns.scatterplot(x=variable1, y=variable2, data=leerDatos, hue=variable)
            st.info("Gráfico de dispersión de las variables:")
            plt.xlabel(variable1)
            plt.ylabel(variable2)
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
        if st.checkbox("Matriz de Correlaciones"):
            st.info("Matriz de correlaciones de tus datos:")
            mCorre=leerDatos.corr(method='pearson')
            st.dataframe(mCorre)
        if st.checkbox("Mapa de Calor"):
            st.info("Mapa de calor de tus datos:")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.figure(figsize=(14,7))
            mCorre=leerDatos.corr(method='pearson')
            mCalor=np.triu(mCorre)
            sns.heatmap(mCorre, cmap="RdBu_r", annot=True, mask=mCalor)
            st.pyplot()
        #Definición de variables predictoras y variable clase
        st.info("Definición de variables predictoras y variable clase")
        st.success("Tu variable clase es  " + variable)
        st.write("Por favor indica el tipo de dato de tu variable clase")
        if st.checkbox("Categóricas"):
           leerDatos["Clase"]=leerDatos[variable]
           varN=leerDatos["Clase"].unique()
           leerDatos=leerDatos.replace({varN[0]:0, varN[1]:1})
           st.write("Estos son tus datos modificados, donde:")
           st.success("0:" + varN[0] + "1:" + varN[1])
           st.dataframe(leerDatos)
           #Número de registros con el mismo valor de la variable
           st.write("Número de registros por grupo:")
           st.dataframe(leerDatos.groupby(variable).size())
           #Variables pedictoras
           try:
               st.info("Selecciona las variables predictoras que deseas incluir en tu modelo:")
               predictoras=st.multiselect("", leerDatos.columns)
               st.write("Tus variables elegidas fueron:")
               st.write(predictoras)
               X=np.array(leerDatos[predictoras])
               st.info("Variables predictoras:")
               st.dataframe(X)
               #Variable clase
               st.info("Variable clase")
               Y=np.array(leerDatos["Clase"])
               st.dataframe(Y)
               #Aplicación del algoritmo
               st.info("Aplicación del algoritmo")
               st.write("Elige el tamaño de tus datos de prueba")
               #Tamaño de prueba
               tamDatos=st.slider("", min_value=0.1, max_value=0.9, step=0.1)
               X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y,test_size = tamDatos,random_state = 1234,shuffle = True)
               #Divsión de datos de prueba y entrenamiento
               st.info("Datos de entrenamiento: Variables predictoras")
               st.dataframe(X_train)
               st.info("Datos de entrenamiento: Variable clase")
               st.dataframe(Y_train)
               #Entrenamiento
               clasificacion=linear_model.LogisticRegression()
               clasificacion.fit(X_train, Y_train)
               #Predicciones probabilísticas de los datos de prueba
               st.info("Datos de prueba: Probabilidades")
               probabilidades=clasificacion.predict_proba(X_validation)
               st.dataframe(probabilidades)
               #Predicciones clasificación final
               st.info("Predicción final")
               predicciones=clasificacion.predict(X_validation)
               st.dataframe(predicciones)
               #Score
               st.success("La exactitud de tu modelo es: " + str(clasificacion.score( X_validation, Y_validation)))
               #Matriz de clasificación
               Y_clasificacion=clasificacion.predict(X_validation)
               matrizClasificacion=pd.crosstab(Y_validation.ravel(),Y_clasificacion,rownames=['Real'],colnames=['Clasificación'])
               st.info("Matriz de clasificación")
               st.write("Columnas: Real")
               st.write("Filas: Clasificación")
               st.dataframe(matrizClasificacion)
               #Reporte de clasificación
               st.info("Reporte de clasificación")
               st.success("Exactitud: " + str(clasificacion.score( X_validation, Y_validation)))
               precision = float(classification_report(Y_validation, Y_clasificacion).split()[10])
               st.success("Precisión: "+ str(precision))
               sensibilidad = float(classification_report(Y_validation, Y_clasificacion).split()[11])
               st.success("Sensibilidad: "+ str(sensibilidad))
               especificidad = float(classification_report(Y_validation, Y_clasificacion).split()[6])
               st.success("Especificidad: "+ str(especificidad))
               st.error("Tasa de error: "+str((1-clasificacion.score(X_validation, Y_validation))))

               # Ecuación del modelo
               st.info("Ecuación modelo")
               st.latex(r"p=\frac{1}{1+e^{-(a+bX)}}")
               st.success("Intercepto: "+str(clasificacion.intercept_[0]))
               st.latex("a+bX="+str(clasificacion.intercept_[0]))
               for i in range(len(predictoras)):
                    predictoras[i] = predictoras[i].replace("_", "")
                    st.latex("+"+str(clasificacion.coef_[0][i].round(3))+"("+str(predictoras[i])+")")

               #Pruebas
               with st.expander("Prueba tu modelo"):
                   st.info("Nuevos pronósticos")
                   nuevosRegistros=[]
                   for i in range(len(predictoras)):
                       nuevosRegistros.append(st.number_input(predictoras[i]))
                   st.success("El pronóstico fue: " + str(clasificacion.predict([nuevosRegistros])))
                   st.write("Interpretación de tus resultados:")
                   st.warning("0:" + varN[0] + "1:" + varN[1])
           except: 
                st.warning("No se ha podido aplicar con exito el algoritmo. Ingrese todos los datos solicitados.")
                    
        if st.checkbox("Númericas"):
           #Variables pedictoras
           try:
               st.info("Selecciona las variables predictoras que deseas incluir en tu modelo:")
               predictoras=st.multiselect("", leerDatos.columns)
               st.write("Tus variables elegidas fueron:")
               st.write(predictoras)
               X=np.array(leerDatos[predictoras])
               st.info("Variables predictoras:")
               st.dataframe(X)
               #Variable clase
               st.info("Variable clase")
               Y=np.array(leerDatos[variable])
               st.dataframe(Y)
               #Aplicación del algoritmo
               st.info("Aplicación del algoritmo")
               st.write("Elige el tamaño de tus datos de prueba")
               #Tamaño de prueba
               tamDatos=st.slider("", min_value=0.1, max_value=0.9, step=0.1)
               X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y,test_size = tamDatos,random_state = 1234,shuffle = True)
               #Divsión de datos de prueba y entrenamiento
               st.info("Datos de entrenamiento: Variables predictoras")
               st.dataframe(X_train)
               st.info("Datos de entrenamiento: Variable clase")
               st.dataframe(Y_train)
               #Entrenamiento
               clasificacion=linear_model.LogisticRegression()
               clasificacion.fit(X_train, Y_train)
               #Predicciones probabilísticas de los datos de prueba
               st.info("Datos de prueba: Probabilidades")
               probabilidades=clasificacion.predict_proba(X_validation)
               st.dataframe(probabilidades)
               #Predicciones clasificación final
               st.info("Predicción final")
               predicciones=clasificacion.predict(X_validation)
               st.dataframe(predicciones)
               #Score
               st.success("La exactitud de tu modelo es: " + str(clasificacion.score( X_validation, Y_validation)))
               #Matriz de clasificación
               Y_clasificacion=clasificacion.predict(X_validation)
               matrizClasificacion=pd.crosstab(Y_validation.ravel(),Y_clasificacion,rownames=['Real'],colnames=['Clasificación'])
               st.info("Matriz de clasificación")
               st.write("Columnas: Real")
               st.write("Filas: Clasificación")
               st.dataframe(matrizClasificacion)
               #Reporte de clasificación
               st.info("Reporte de clasificación")
               st.success("Exactitud: " + str(clasificacion.score( X_validation, Y_validation)))
               precision = float(classification_report(Y_validation, Y_clasificacion).split()[10])
               st.success("Precisión: "+ str(precision))
               sensibilidad = float(classification_report(Y_validation, Y_clasificacion).split()[11])
               st.success("Sensibilidad: "+ str(sensibilidad))
               especificidad = float(classification_report(Y_validation, Y_clasificacion).split()[6])
               st.success("Especificidad: "+ str(especificidad))
               st.error("Tasa de error: "+str((1-clasificacion.score(X_validation, Y_validation))))

               # Ecuación del modelo
               st.info("Ecuación modelo")
               st.latex(r"p=\frac{1}{1+e^{-(a+bX)}}")
               st.success("Intercepto: "+str(clasificacion.intercept_[0]))
               st.latex("a+bX="+str(clasificacion.intercept_[0]))
               for i in range(len(predictoras)):
                    predictoras[i] = predictoras[i].replace("_", "")
                    st.latex("+"+str(clasificacion.coef_[0][i].round(3))+"("+str(predictoras[i])+")")

               #Pruebas
               with st.expander("Prueba tu modelo"):
                   st.info("Nuevos pronósticos")
                   nuevosRegistros=[]
                   for i in range(len(predictoras)):
                       nuevosRegistros.append(st.number_input(predictoras[i]))
                   st.success("El pronóstico fue: " + str(clasificacion.predict([nuevosRegistros])))
           except: 
                st.warning("No se ha podido aplicar con exito el algoritmo. Ingrese todos los datos solicitados.")
                
if option == "Árbol de Decisión-Pronóstico": 
    st.title("Árbol de Decisión-Pronóstico")
    #Cargar datos
    datosArchivo=st.file_uploader("Selecciona un archivo", type=["csv"])
    st.warning("Por favor sube un archivo .CSV")
    if datosArchivo is not None:
        leerDatos=pd.read_csv(datosArchivo)
        st.info("Visualización de tus datos")
        st.dataframe(leerDatos)
        #Descipción de los datos
        st.info("Descripción de los datos")
        st.dataframe(leerDatos.describe())
        #Gáfica de los datos
        st.info("Grafica de los datos")
        st.write("Selecciona tus variables a observar")
        variable1=st.selectbox("Eje X:", leerDatos.columns)
        variable2=st.selectbox("Eje Y:", leerDatos.columns)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.figure(figsize=(10,5))
        plt.plot(leerDatos[variable1],leerDatos[variable2], color='green', marker='o', label=variable2)
        plt.xlabel(variable1)
        plt.ylabel(variable2)
        plt.title("Tus datos")
        plt.grid(True)
        plt.legend()
        st.pyplot()
        #Selección de características	
        st.info("Selección de características")
        #Elegir la variable a trabajar 
        st.warning("Selecciona la variable para trabajar:")
        vari=st.selectbox("", leerDatos.columns)
        st.write("Elige la forma en que quieres realizar tu selección de carácterísticas:")
        if st.checkbox("Gráfico de dispersión"):
            st.write("Elige 2 variables")
            vari1=st.selectbox("Variable 1:", leerDatos.columns)
            vari2=st.selectbox("Variable 2:", leerDatos.columns)
            sns.scatterplot(x=vari1, y=vari2, data=leerDatos, hue=vari)
            st.write("Gráfico de dispersión de las variables:")
            plt.xlabel(vari1)
            plt.ylabel(vari2)
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
        if st.checkbox("Matriz de correlaciones"):
            st.write("Matriz de correlaciones de tus datos:")
            mCorre=leerDatos.corr(method='pearson')
            st.dataframe(mCorre)
        if st.checkbox("Mapa de calor"):
            st.write("Mapa de calor de tus datos:")
            plt.figure(figsize=(14,7))
            mCorre=leerDatos.corr(method='pearson')
            mCalor=np.triu(mCorre)
            sns.heatmap(mCorre, cmap="RdBu_r", annot=True, mask=mCalor)
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
        #Definición de variables predictoras y variable clase
        st.info("Definición de variables predictoras y variable pronóstico")
        st.success("Tu variable pronóstico es  " + vari)
        try:
            st.info("Selecciona las variables predictoras que deseas incluir en tu modelo:")
            predictoras=st.multiselect("", leerDatos.columns)
            st.write("Tus variables elegidas fueron:")
            st.write(predictoras)
            X=np.array(leerDatos[predictoras])
            st.info("Variables predictoras:")
            st.dataframe(X)
            #Variable pronóstico
            st.info("Variable pronóstico")
            Y=np.array(leerDatos[vari])
            st.dataframe(Y)
            #Aplicación del algoritmo
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            from sklearn import model_selection
            st.info("Aplicación del algoritmo")
            st.write("Elige el tamaño de tus datos de prueba")
            #Tamaño de prueba
            tamDatos=st.slider("", min_value=0.1, max_value=0.9, step=0.1)
            X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y,test_size = tamDatos,random_state = 1234,shuffle = True)
            #Divsión de datos de prueba y entrenamiento
            st.info("Datos de entrenamiento: Variables predictoras")
            st.dataframe(X_train)
            st.info("Datos de entrenamiento: Variable pronóstico")
            st.dataframe(Y_train)
            #Entrenamiento
            st.info("Entrenamiento del modelo")
            st.write('Ingresa tus parámetros para construir el árbol')
            #Parámetros
            maxDepth=st.number_input("Máximo de profundidad:")
            minSamplesSplit=st.number_input("Mínimo de muestras para dividir:")
            minSamplesLeaf=st.number_input("Mínimo de muestras en hoja:")
            pronosticoAP=DecisionTreeRegressor(max_depth=int(maxDepth), min_samples_split=int(minSamplesSplit), min_samples_leaf=int(minSamplesLeaf))
            pronosticoAP.fit(X_train, Y_train)
            st.info("Pronóstico")
            Y_pred=pronosticoAP.predict(X_test)
            val=pd.DataFrame(Y_test,Y_pred)
            st.dataframe(val)
            st.info("Gráfica pronóstico")
            plt.figure(figsize=(10,5))
            plt.plot(Y_test,color='green', marker='o', label='Y_test')
            plt.plot(Y_pred,color='red', marker='o', label='Y_pred')
            plt.xlabel(variable1)
            plt.ylabel(variable2)
            plt.title("Tus datos")
            plt.grid(True)
            plt.legend()
            st.pyplot()
            #Obtención de los parámetros del modelo
            st.info("Parámetros del modelo")
            st.write("El modelo tiene los siguientes parámetros:")
            st.success("La exactitud de tu modelo es: " + str(r2_score( Y_test, Y_pred)))
            st.success('Criterio:' +  str(pronosticoAP.criterion))
            #st.success('Importancia variables:' + str(pronosticoAP.feature_importances_))
            st.success('MAE: %.4f' % mean_absolute_error(Y_test, Y_pred))
            st.success('MSE: %.4f' % mean_squared_error(Y_test, Y_pred))
            st.success("RMSE: %.4f" % mean_squared_error(Y_test, Y_pred, squared=False))
            #Importancia de variables
            st.info("Importancia de variables")
            st.write("Las variables más importantes son:")
            importancia=pd.DataFrame({'Variables':predictoras,'Importancia':pronosticoAP.feature_importances_}).sort_values(by='Importancia', ascending=False)
            st.dataframe(importancia)
            #Árbol de decisión
            st.info("Árbol de decisión")
            import graphviz
            from sklearn.tree import export_graphviz
            elementos = export_graphviz(pronosticoAP,feature_names=predictoras)
            arbolP=graphviz.Source(elementos)
            arbolP.format = 'svg'
            arbolP.render('ÁrbolDecisiónPronóstico')
            #Descarga del árbol
            with open("ÁrbolDecisiónPronóstico.svg", "rb") as file:
                        btn = st.download_button(label="Descarga tu Árbol de Decisión (Pronóstico)",data=file,file_name="ÁrbolDecisiónPronóstico.svg",mime="image/svg")
            #Pruebas
            with st.expander("Prueba tu modelo"):
                st.info("Nuevos pronósticos")
                nuevosRegistros=[]
                for i in range(len(predictoras)):
                    nuevosRegistros.append(st.number_input(predictoras[i]))
                st.success("El pronóstico fue: " + str(pronosticoAP.predict([nuevosRegistros])))
        except:
            st.warning("No se ha podido aplicar con exito el algoritmo. Ingrese todos los datos solicitados.")

if option == "Árbol de Decisión-Clasificación":
    st.title("Árbol de Decisión-Clasificación")
    #Cargar datos
    datosArchivo=st.file_uploader("Selecciona un archivo", type=["csv"])
    st.warning("Por favor sube un archivo .CSV")
    if datosArchivo is not None:
        leerDatos=pd.read_csv(datosArchivo)
        st.info("Visualización de tus datos")
        st.dataframe(leerDatos)
        #Elegir la variable a trabajar 
        st.warning("Selecciona la variable para trabajar:")
        variC=st.selectbox("", leerDatos.columns)
        #Número de registros con el mismo valor de la variable
        st.write("Número de registros por grupo:")
        st.dataframe(leerDatos.groupby(variC).size())
        #Selección de características	
        st.info("Selección de características")
        st.write("Elige la forma en que quieres realizar tu selección de carácterísticas:")
        if st.checkbox("Gráfico de dispersión"):
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write("Elige 2 variables")
            vari1C=st.selectbox("Variable 1:", leerDatos.columns)
            vari2C=st.selectbox("Variable 2:", leerDatos.columns)
            sns.scatterplot(x=vari1C, y=vari2C, data=leerDatos, hue=vari1C)
            st.write("Gráfico de dispersión de las variables:")
            plt.xlabel(vari1C)
            plt.ylabel(vari2C)
            st.pyplot()
        if st.checkbox("Matriz de correlaciones"):
            st.write("Matriz de correlaciones de tus datos:")
            mCorre=leerDatos.corr(method='pearson')
            st.dataframe(mCorre)
        if st.checkbox("Mapa de calor"):
            st.write("Mapa de calor de tus datos:")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.figure(figsize=(14,7))
            mCorre=leerDatos.corr(method='pearson')
            mCalor=np.triu(mCorre)
            sns.heatmap(mCorre, cmap="RdBu_r", annot=True, mask=mCalor)
            st.pyplot()
        #Definición de variables predictoras y variable clase
        st.info("Definición de variables predictoras y variable clase")
        st.success("Tu variable clase es  " + variC)
        try:
            st.info("Selecciona las variables predictoras que deseas incluir en tu modelo:")
            predictoras=st.multiselect("", leerDatos.columns)
            st.write("Tus variables elegidas fueron:")
            st.write(predictoras)
            X=np.array(leerDatos[predictoras])
            st.info("Variables predictoras")
            st.dataframe(X)
            #Variable clase
            st.info("Variable clase")
            Y=np.array(leerDatos[variC])
            st.dataframe(Y)
            #Aplicación del algoritmo
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.metrics import classification_report
            from sklearn.metrics import confusion_matrix
            from sklearn.metrics import accuracy_score
            from sklearn import model_selection
            st.info("Aplicación del algoritmo")
            st.write("Elige el tamaño de tus datos de prueba")
            #Tamaño de prueba
            tamDatos=st.slider("", min_value=0.1, max_value=0.9, step=0.1)
            X_train, X_validation,Y_train, Y_validation= model_selection.train_test_split(X, Y,test_size = tamDatos,random_state = 0,shuffle = True)
            #Divsión de datos de prueba y entrenamiento
            st.info("Datos de entrenamiento: Variables predictoras")
            st.dataframe(X_train)
            st.info("Datos de entrenamiento: Variable clase")
            st.dataframe(Y_train)
            #Entrenamiento
            st.info("Entrenamiento del modelo")
            st.write('Ingresa tus parámetros para construir el árbol')
            #Parámetros
            maxDepthC=st.number_input("Máximo de profundidad:")
            minSamplesSplitC=st.number_input("Mínimo de muestras para dividir:")
            minSamplesLeafC=st.number_input("Mínimo de muestras en hoja:")
            clasificacionAD=DecisionTreeClassifier(random_state=0, max_depth=int(maxDepthC), min_samples_leaf=int(minSamplesLeafC), min_samples_split=int(minSamplesSplitC))
            clasificacionAD.fit(X_train, Y_train)
            #Clasificaciones
            st.info("Clasificaciones")
            Y_clasificacion=clasificacionAD.predict(X_validation)
            valores=pd.DataFrame(Y_validation,Y_clasificacion)
            st.dataframe(valores)
            #Validación del modelo
            #Score
            st.success("La exactitud de tu modelo es: " + str(clasificacionAD.score( X_validation, Y_validation)))
            #Matriz de confusión
            st.info("Matriz de clasificación")
            Y_clasificacion=clasificacionAD.predict(X_validation)
            matrizC=pd.crosstab(Y_validation.ravel(), Y_clasificacion, rownames=['Real'], colnames=['Clasificación'])
            st.write("Columnas: Real")
            st.write("Filas: Clasificación")
            st.dataframe(matrizC)
            #Reporte de clasificación
            st.info("Reporte de clasificación")
            st.success("Exactitud: " + str(clasificacionAD.score( X_validation, Y_validation)))
            precision = float(classification_report(Y_validation, Y_clasificacion).split()[10])
            st.success("Precisión: "+ str(precision))
            sensibilidad = float(classification_report(Y_validation, Y_clasificacion).split()[11])
            st.success("Sensibilidad: "+ str(sensibilidad))
            especificidad = float(classification_report(Y_validation, Y_clasificacion).split()[6])
            st.success("Especificidad: "+ str(especificidad))
            st.error("Tasa de error: "+str((1-clasificacionAD.score(X_validation, Y_validation))))
            #Importancia de variables
            st.info("Importancia de variables")
            importancia=pd.DataFrame({'Variables':predictoras,'Importancia':clasificacionAD.feature_importances_}).sort_values(by='Importancia', ascending=False)
            st.dataframe(importancia)
            #Árbol de decisión
            import graphviz
            from sklearn.tree import export_graphviz
            st.info("Árbol de decisión")
            elementos=export_graphviz(clasificacionAD,feature_names=predictoras,class_names=Y_clasificacion)
            arbolC=graphviz.Source(elementos)
            arbolC.format = 'svg'
            arbolC.render('ÁrbolDecisiónClasificación')
            #Descarga del árbol
            with open("ÁrbolDecisiónClasificación.svg", "rb") as file:
                        btn = st.download_button(label="Descarga tu Árbol de Decisión (Clasificación)",data=file,file_name="ÁrbolDecisiónClasificación.svg",mime="image/svg")
            #Pruebas
            with st.expander("Prueba tu modelo"):
                st.info("Nuevos pronósticos")
                nuevosRegistros=[]
                for i in range(len(predictoras)):
                    nuevosRegistros.append(st.number_input(predictoras[i]))
                st.success("El pronóstico fue: " + str(clasificacionAD.predict([nuevosRegistros])))
        except:
            st.warning("No se ha podido aplicar con exito el algoritmo. Ingrese todos los datos solicitados.")
    





 
