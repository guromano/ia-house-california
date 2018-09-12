import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

def ConverterFloat(data):
    datac = pd.to_numeric(data, errors='coerce')
    return datac

def ConverterInt(data):
    datac = pd.to_numeric(data, errors='coerce',downcast='integer')
    return datac

# carregar arquivo OK
dados = pd.read_csv('precos_casa_california.csv')

# tratar dados OK
dadost = dados
dadost["longitude"] = ConverterFloat(dadost["longitude"])
dadost["latitude"] = ConverterFloat(dadost["latitude"])
dadost["housing_median_age"] = ConverterInt(dadost["housing_median_age"])
dadost["total_rooms"] = ConverterInt(dadost["total_rooms"])
dadost["total_bedrooms"] = ConverterInt(dadost["total_bedrooms"])
dadost["population"] = ConverterInt(dadost["population"])
dadost["households"] = ConverterInt(dadost["households"])
dadost["median_income"] = ConverterFloat(dadost["median_income"])
dadost["median_house_value"] = ConverterFloat(dadost["median_house_value"])
dadost['ocean_proximity'] = dadost['ocean_proximity'].replace({'<1H OCEAN': 0, 'INLAND': 1,'NEAR OCEAN':2,'NEAR BAY':3, 'ISLAND':4})
dadost["ocean_proximity"] = ConverterFloat(dadost["ocean_proximity"])
dadost = dadost.dropna()
print(dadost)

# OK Adicionar e remover conteudo das colunas
dadosp = dadost
dadosp['rooms_p_household'] = dadosp['total_rooms'] / dadosp['households']
dadosp['bedrooms_p_rooms']= dadosp['total_bedrooms']/dadosp['total_rooms']
dadosp['population_p_household'] = dadosp['population']/dadosp['households']

dadosp = dadosp[dadosp['median_house_value']<=1000000].reset_index(drop=True)
dadosp = dadosp[dadosp['housing_median_age']<=100].reset_index(drop=True)
dadosp=dadosp[dadosp['median_income']<=11].reset_index(drop=True)
dadosp=dadosp[dadosp['population']<=5000].reset_index(drop=True)
dadosp=dadosp[dadosp['bedrooms_p_rooms']<=0.4].reset_index(drop=True)
dadosp=dadosp[(dadosp['population_p_household']>=1) & (dadosp['population_p_household']<7)].reset_index(drop=True)
dadosp=dadosp[((dadosp['median_house_value']>=400000) & (dadosp['median_income']>=6)) | ((dadosp['median_house_value']<400000) & (dadosp['median_income']<6))].reset_index(drop=True)
dadosp=dadosp[dadosp['rooms_p_household']<20]

corr_matrix = dadosp.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
dadosp.hist(bins=55,figsize=(17,12))
plt.show()

# OK Separar em treino e teste
y = dadosp['median_house_value']
x = dadosp.drop(columns=['median_house_value'])
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.2,random_state = 7)

# OK Treinar modelo

# Trocando para esse metodo teve pior desempenho
# model = LinearRegression()
model = GradientBoostingRegressor()
model.fit(x_treino,y_treino)


# OK Testar modelo
score = model.score(x_teste,y_teste)
print(score)

