# Simulações de Monte Carlo para Modelagem Financeira

Nesse projeto vamos explorar a paralelização de simulações Monte Carlo para modelagem financeira usando CUDA.

# Introdução

Simulações de Monte Carlo são extremamente relevantes no mundo estatístico por possibilitarem a inclusão de um elemento de incerteza ou aleatoriedade nas predições. Usaremos CUDA para acelerar simulações de Monte Carlo usadas em modelos financeiros, como precificação de opções. Tentaremos gerar amostras aleatórias e calcular estimativas estatísticas em paralelo.

# Simulações Monte Carlo

Simulações Monte Carlo permitem que a gente modele sistemas estocásticos complexos, usando experimentos aleatórios e coletando dados sobre as saídas. Usando este método, podemos estimar preços e outros valores que não seriam possíveis calcular diretamente. Fazendo a migração para a GPU usando C++ e CUDA, teremos uma perfomance melhor, maior throughout e maior escalabilidade.

# Algoritmo de Precificação

- Inicialização da parâmetros e variáveis
- Geração de números aleatórios
- Simulação de caminhos aleatórios
- Cálculo da média

# Barrier Call Option

Para o projeto, faremos a implementação usando barrier call options. Options são um instrumento financeiro que dá o direito da pessoa comprar uma ação, apenas se a ação não ultrapassar uma barreira de preço e estiver acima do strike price. No exemplo da esquerda, a ação não ultrapassa a barreira e fica acima do strike, portanto podemos comprar a ação por 90 dólares, mesmo ela valendo 115 dólares no vencimento. No exemplo da direita, mesmo ela estando acima do strike, ela passou a barreira e portanto a option expira.
![image](https://github.com/mc970-24s1/projeto-final-monte-carlo-simulations/assets/120742067/9dff47dd-26d6-4b4a-9345-c1c3acd5b1db)

# [Black-Scholes Model Formula](https://www.investopedia.com/terms/b/blackscholes.asp)

Usaremos uma derivação do Black-Scholes Model, por ser um dos melhores métodos de precificação dos contratos de options. Não entraremos afundo na parte matemática, mas alguns parâmetros são usados pra calcular o preço futuro de uma ação:

- Preço atual
- Taxa de juros
- Volatilidade
- Maturidade
- Números aleatórios seguindo uma distribuição normal (razão da simulação Monte Carlo)

# Implementações

Primeiro faremos a implementação serial na CPU usando C++. Depois implementaremos o algortimo paralelo usando CUDA.

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/IwHO6ydp)
