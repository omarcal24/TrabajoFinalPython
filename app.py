from time import time
import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
from IPython.display import display  # Allows the use of display() for DataFrames
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost.sklearn import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

st.set_page_config(page_title="Trabajo Final de Máster por Adolfo Omar Calderón",
                   layout='wide')

st.title("Trabajo Final para el Máster en Programación Avanzada con Python para Big Data, Machine Learning y Hacking")

st.header("Trabajo hecho por Adolfo Omar Calderón")
st.header("Modelo Predictivo y Análisis de Datos")

st.header("Análisis Exploratorio de Datos")
df = pd.read_csv('ford.csv')

st.subheader("Primeras columnas del dataset")
st.table(df.head())

st.subheader("Descripción del dataset")
st.table(df.describe())

st.subheader("Valores únicos por columna")

with st.echo():
    columns_list = df.columns.values.tolist()
    for column in columns_list:
        print(column)
        print(df[column].unique())
        print('\n')

col_dict = {}

columns_list = df.columns.values.tolist()
for column in columns_list:
    col_dict[column] = df[column].unique()

#st.write(col_dict)

my_expander = st.beta_expander("Valores únicos por columna", expanded=False)

with my_expander:
    for key, value in col_dict.items():
        st.write(key, value)

st.subheader(
    "Corrijo valores en model con typos de espacio")
with st.echo():
    df['model'] = df['model'].str.strip()

st.subheader("Drop de una fila que no es necesaria y es un outlier por tener año 2060")
st.write(df[df['year'] == 2060])
#if st.button("Drop a las columnas"):
with st.echo():
    df.drop([17726], axis=0, inplace=True)

st.subheader(
    "Valores nulos")
st.write(df.isna().sum())


st.subheader(
    "Boxplot de precio de los coches y tipo de transmisión")
fig1 = plt.figure(figsize=(14, 4))
sns.boxplot(df['transmission'], df['price'])
st.pyplot(fig1)

st.subheader(
    "Boxplot de precio de los coches y tipo de combustible que utilizan")
fig2 = plt.figure(figsize=(14, 4))
sns.boxplot(df['fuelType'], df['price'])
st.pyplot(fig2)

st.subheader(
    "Scatterplot de precio de los coches y año")
fig3 = plt.figure(figsize=(14, 4))
sns.scatterplot(df['year'], df['price'])
st.pyplot(fig3)

st.subheader(
    "Scatterplot de precio de los coches y cantidad de millas recorridas")
fig4 = plt.figure(figsize=(14, 4))
sns.scatterplot(df['mileage'], df['price'])
st.pyplot(fig4)

st.subheader(
    "Scatterplot de precio de los coches y rendimiento en galones de galones por millas")
fig5 = plt.figure(figsize=(14, 4))
sns.scatterplot(df['mpg'], df['price'])
st.pyplot(fig5)


def column_percentage(data, column):
    column_values = data[column].value_counts(normalize=True)
    column_values = pd.DataFrame(column_values)
    column_values = column_values.reset_index()
    column_values = column_values.rename(
        columns={'index': column, column: 'Porcentage de autos'})
    column_values['Porcentage de autos'] = np.round(
        column_values['Porcentage de autos'] * 100, 1)

    return column_values

model_perc = column_percentage(df, 'model')

st.subheader(
    "Barplot de porcentaje de coches por modelo")
fig6 = plt.figure(figsize=(14,4))
sns.barplot(x = 'model', y = 'Porcentage de autos', data=model_perc)
plt.xticks(rotation = 90)
st.pyplot(fig6)

fuel_perc = column_percentage(df, 'fuelType')

st.subheader(
    "Barplot de porcentaje de coches por tipo de combustible")
fig7 = plt.figure(figsize=(14, 4))
sns.barplot(x='fuelType', y='Porcentage de autos', data=fuel_perc)
plt.xticks(rotation=90)
st.pyplot(fig7)

year_perc = column_percentage(df, 'year')

st.subheader(
    "Barplot de porcentaje de coches por su año")
fig8 = plt.figure(figsize=(14, 4))
sns.barplot(x='year', y='Porcentage de autos', data=year_perc)
plt.xticks(rotation=90)
st.pyplot(fig8)

transmission_perc = column_percentage(df, 'transmission')

st.subheader(
    "Barplot de porcentaje de coches por tipo de transmisión")
fig9 = plt.figure(figsize=(14, 4))
sns.barplot(x='transmission', y='Porcentage de autos', data=transmission_perc)
plt.xticks(rotation=90)
st.pyplot(fig9)


st.header("Machine Learning")
st.header("Aprendizaje Supervisado")


st.subheader("Variables dependiente e independientes")
with st.echo():
    y = df['price']
    X = df.drop('price', axis=1)


st.subheader("Variables dummies")
with st.echo():
    X_dummies = pd.get_dummies(X, drop_first=True)

st.table(X_dummies.head())


st.subheader("Train test split")
with st.echo():
    X_train, X_test, y_train, y_test = train_test_split(X_dummies,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)


def train_predict(learner, X_train, y_train, X_test, y_test):

    results = {}
    imp = {}

    start = time()
    learner = learner.fit(X_train, y_train)
    end = time()

    results['train_time'] = end - start

    start = time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train)
    end = time()

    results['pred_time'] = end - start

    results['mean_absolute_error'] = np.round(mean_absolute_error(
        y_train, predictions_train), 2)

    results['mean_squared_error'] = np.round(mean_squared_error(
        y_train, predictions_train), 2)

    results['root_mean_squared_error'] = np.round(mean_squared_error(
        y_train, predictions_train, squared=False), 2)

    results['r2_score'] = np.round(r2_score(y_test, predictions_test), 2)

    #imp[learner.__class__.__name__] = list(zip(learner.feature_importances_, X_train.columns))

    print("{} trained.".format(learner.__class__.__name__))

    return results


st.subheader("Selección de tres modelos, Linear Regression, XGBRegressor y Decision Tree Regressos")
with st.echo():
    reg_A = LinearRegression()
    reg_B = XGBRegressor()
    reg_C = DecisionTreeRegressor()

    results = {}
    for reg in [reg_A, reg_B, reg_C]:
        reg_name = reg.__class__.__name__
        results[reg_name] = {}
        results[reg_name][0] = \
            train_predict(reg, X_train, y_train, X_test, y_test)


    results = results

for i, ii in results.items():
    st.write(i, ii)
    st.write('')

st.subheader("El mejor modelo es el Decision Tree Regressor asi que es con el que trabajaré")

st.subheader("Selecciono el modelo")
with st.echo():
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

st.subheader(
    "Utilizo un gran atributo de este modelo que es su importancia de cada feature")

feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
fig10 = plt.figure(figsize=(14, 4))
feat_importances.nlargest(20).plot(kind='barh')
st.pyplot(fig10)

st.header(X_dummies.shape)
st.table(X_dummies.head())

with st.form(key='my_form'):
    year = st.selectbox('Año del carro', (range(1996, 2021)))
    millas = st.text_input("Cantidad de millas del coche")
    mpg = st.text_input("Rendimiento de millas por galón")
    motor = st.text_input('Tamaño del motor, usar puntos y no comas')
    modelo = st.selectbox('Modelo del carro', ('C-MAX', 'EcoSport', 'Edge', 'Escort', 'Fiesta', 'Focus', 'Fusion', 'Galaxy', 'Grand C-Max', 'Grand Tourneo Conect',
                                               'KA', 'KA+', 'Kuga', 'Mondeo', 'Mustang', 'Puma', 'Ranger', 'S-MAX', 'Steeka', 'Tourneo Connect',
                                               'Tourneo Custom', 'Transit Tourneo'))
    transmision = st.selectbox('Tipo de transmisión', ('Manual', 'Semi-Auto', 'Automática'))
    fuel = st.selectbox('Tipo de combustible', ('Eléctrico', 'Hybrid', 'Otro', 'Petrol'))
    submit_button = st.form_submit_button(label='Submit')



modelos = ['C-MAX', 'EcoSport', 'Edge', 'Escort', 'Fiesta', 'Focus', 'Fusion', 'Galaxy', 'Grand C-Max', 'Grand Tourneo Conect',
           'KA', 'KA+', 'Kuga', 'Mondeo', 'Mustang', 'Puma', 'Ranger', 'S-MAX', 'Steeka', 'Tourneo Connect'
           'Tourneo Custom', 'Transit Tourneo']

carro = [0] * 33
if transmision == 'Manual':
    carro[27] = 1
if transmision == 'Semi-Auto':
    carro[28] = 1

if fuel == 'Eléctrico':
    carro[29] = 1
if fuel == 'Hybrid':
    carro[30] = 1
if fuel == 'Otro':
    carro[31] = 1
if fuel == 'Petrol':
    carro[32] = 1

indice_modelo = modelos.index(modelo) + 4

carro[0] = year
carro[1] = millas
carro[2] = 150
carro[3] = motor
carro[indice_modelo] = 1


carro = [carro]

try:
    if type(carro[0]) != str:
        st.subheader("Predicción del precio en el mercado del coche")
        #carro[1] = int(carro[1])
        #carro[3] = float(carro[3])
        
        precio_predecido = model.predict(carro)
        st.write("El precio predecido para vender este modelo Ford de segunda es: ")
        st.write(precio_predecido[0], "$")
except:
    pass
