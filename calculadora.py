import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
except ModuleNotFoundError:
    print("TensorFlow não está instalado. Por favor, instale usando: pip install tensorflow")
    tensorflow_available = False
else:
    tensorflow_available = True

# Verifica se o arquivo existe antes de carregar
file_path = r"C:\Users\felip\OneDrive\Área de Trabalho\Aiube-2025\house_prices.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado. Verifique o caminho e tente novamente.")

# Carregar os dados
base = pd.read_csv(file_path)
base = base[['price', 'sqft_living', 'sqft_lot', 'floors']].dropna().tail(200)

# Normalização dos dados
scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(base), columns=base.columns)

# Divisão dos dados
train_data, test_data = train_test_split(data_scaled, test_size=0.3, random_state=1)
X_train, y_train = train_data.drop(columns=['price']), train_data['price']
X_test, y_test = test_data.drop(columns=['price']), test_data['price']

# Desnormalizando os valores reais de y_test
y_test_real = y_test * (base['price'].max() - base['price'].min()) + base['price'].min()

if tensorflow_available:
    # Construção da rede neural
    model = Sequential([
        Dense(5, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(5, activation='relu'),  # Adicionando uma segunda camada oculta
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Treinamento da rede neural
    history = model.fit(X_train, y_train, epochs=100, verbose=0, validation_data=(X_test, y_test))

    # Plot da função de perda
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Previsões da rede neural
    y_pred_nn = model.predict(X_test).flatten()

    # Desnormalizando previsões
    y_pred_nn_real = y_pred_nn * (base['price'].max() - base['price'].min()) + base['price'].min()

    # Cálculo do erro quadrático médio (MSE)
    mse_nn = np.mean((y_test_real - y_pred_nn_real) ** 2)
    print(f'MSE Neural Network: {mse_nn}')
else:
    mse_nn = float('inf')
    print("TensorFlow não disponível. Pulando a execução da rede neural.")

# Modelo de regressão linear
lm = LinearRegression()
lm.fit(X_train, y_train)

# Previsões do modelo de regressão
y_pred_lm = lm.predict(X_test)

# Desnormalizando previsões
y_pred_lm_real = y_pred_lm * (base['price'].max() - base['price'].min()) + base['price'].min()

# Cálculo do erro quadrático médio (MSE)
mse_lm = np.mean((y_test_real - y_pred_lm_real) ** 2)
print(f'MSE Regressão Linear: {mse_lm}')

# Comparação de erros
print(f'Melhor modelo: {"Rede Neural" if mse_nn < mse_lm else "Regressão Linear"}')
