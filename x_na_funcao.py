import random
import numpy as np

# Função objetivo: f(x) = x^3 - 6x + 14
def funcao_objetivo(x):
    return x**3 - 6*x + 14

# Converte um vetor binário em um número real na faixa [-10, 10]
def binario_para_real(binario, limite_inferior, limite_superior):
    # Converte o vetor binário em um inteiro
    inteiro = int("".join(str(bit) for bit in binario), 2)
    max_inteiro = (2 ** len(binario)) - 1  # Maior valor possível com os bits
    return limite_inferior + (inteiro / max_inteiro) * (limite_superior - limite_inferior)

# Gera um cromossomo aleatório (binário)
def gerar_cromossomo(tamanho):
    return [random.randint(0, 1) for _ in range(tamanho)]

# Função para criar a população inicial
def criar_populacao(tamanho_populacao, tamanho_cromossomo):
    return [gerar_cromossomo(tamanho_cromossomo) for _ in range(tamanho_populacao)]

# Função de mutação com taxa de mutação configurável
def mutacao(cromossomo, taxa_mutacao):
    for i in range(len(cromossomo)):
        if random.random() < taxa_mutacao:
            cromossomo[i] = 1 - cromossomo[i]

# Função de crossover (1 ou 2 pontos de corte)
def crossover(pai1, pai2, pontos_corte=1):
    if pontos_corte == 1:
        ponto_corte = random.randint(1, len(pai1) - 1)
        filho1 = pai1[:ponto_corte] + pai2[ponto_corte:]
        filho2 = pai2[:ponto_corte] + pai1[ponto_corte:]
    elif pontos_corte == 2:
        ponto1 = random.randint(1, len(pai1) - 2)
        ponto2 = random.randint(ponto1 + 1, len(pai1) - 1)
        filho1 = pai1[:ponto1] + pai2[ponto1:ponto2] + pai1[ponto2:]
        filho2 = pai2[:ponto1] + pai1[ponto1:ponto2] + pai2[ponto2:]
    return filho1, filho2

# Função de seleção por torneio
def selecao_torneio(populacao, fitness_populacao, tamanho_torneio=3):
    torneio = random.sample(list(zip(fitness_populacao, populacao)), tamanho_torneio)
    vencedor = min(torneio, key=lambda x: x[0])  # Seleciona o cromossomo com o menor fitness (mínimo)
    return vencedor[1]

# Função de seleção por roleta viciada
def selecao_roleta_viciada(populacao, fitness_populacao):
    # Para minimizar, converter fitness em valores positivos
    max_fitness = max(fitness_populacao)
    fitness_ajustado = [(max_fitness - f) for f in fitness_populacao]

    # Evitar casos em que todos os fitness são iguais, resultando em probabilidades inválidas
    soma_fitness = sum(fitness_ajustado)
    if soma_fitness == 0:
        probabilidades = [1 / len(fitness_ajustado)] * len(fitness_ajustado)  # Distribuir uniformemente
    else:
        probabilidades = [f / soma_fitness for f in fitness_ajustado]

    return random.choices(populacao, weights=probabilidades, k=1)[0]

# Função principal do algoritmo genético
def algoritmo_genetico(tamanho_populacao, tamanho_cromossomo, geracoes_max, taxa_mutacao=0.01, pontos_corte=1, 
                       metodo_selecao="torneio", elitismo=True, percentual_elitismo=0.1):
    # Faixa de valores de x [-10, 10]
    limite_inferior, limite_superior = -10, 10

    # Criando a população inicial
    populacao = criar_populacao(tamanho_populacao, tamanho_cromossomo)

    for geracao in range(geracoes_max):
        # Avaliando o fitness de cada indivíduo
        fitness_populacao = [funcao_objetivo(binario_para_real(cromossomo, limite_inferior, limite_superior)) for cromossomo in populacao]

        # Encontrando o melhor indivíduo da geração
        melhor_fitness_geracao = min(fitness_populacao)
        melhor_individuo_geracao = populacao[np.argmin(fitness_populacao)]
        melhor_x_geracao = binario_para_real(melhor_individuo_geracao, limite_inferior, limite_superior)

        # Exibindo os resultados da geração
        print(f"Geração {geracao + 1}: Melhor x: {melhor_x_geracao:.5f}, Melhor valor da função: {melhor_fitness_geracao:.5f}")

        # Se usar elitismo, separar os melhores indivíduos
        if elitismo:
            numero_elitismo = int(percentual_elitismo * tamanho_populacao)
            elite = sorted(zip(fitness_populacao, populacao))[:numero_elitismo]
            elite = [individuo for _, individuo in elite]

        nova_populacao = []
        while len(nova_populacao) < tamanho_populacao - (numero_elitismo if elitismo else 0):
            # Seleção dos pais
            if metodo_selecao == "torneio":
                pai1 = selecao_torneio(populacao, fitness_populacao)
                pai2 = selecao_torneio(populacao, fitness_populacao)
            elif metodo_selecao == "roleta":
                pai1 = selecao_roleta_viciada(populacao, fitness_populacao)
                pai2 = selecao_roleta_viciada(populacao, fitness_populacao)

            # Crossover
            filho1, filho2 = crossover(pai1, pai2, pontos_corte)
            nova_populacao.extend([filho1, filho2])

        # Aplicando mutação
        for individuo in nova_populacao:
            mutacao(individuo, taxa_mutacao)

        # Adicionando elite (se existir)
        if elitismo:
            nova_populacao.extend(elite)

        populacao = nova_populacao

    # Encontrando o melhor indivíduo ao final do algoritmo
    fitness_populacao = [funcao_objetivo(binario_para_real(cromossomo, limite_inferior, limite_superior)) for cromossomo in populacao]
    melhor_individuo = populacao[np.argmin(fitness_populacao)]
    melhor_x = binario_para_real(melhor_individuo, limite_inferior, limite_superior)
    melhor_fitness = min(fitness_populacao)

    return melhor_x, melhor_fitness

# Parâmetros de exemplo
tamanho_populacao = 10
tamanho_cromossomo = 10  # Para a precisão, pode ajustar o tamanho dos cromossomos
geracoes_max = 50
taxa_mutacao = 0.01
pontos_corte = 2
metodo_selecao = "torneio"
elitismo = True
percentual_elitismo = 0.1

# Executando o algoritmo genético
melhor_x, melhor_fitness = algoritmo_genetico(tamanho_populacao, tamanho_cromossomo, geracoes_max, 
                                              taxa_mutacao, pontos_corte, metodo_selecao, elitismo, percentual_elitismo)

print(f"\nMelhor x encontrado: {melhor_x}")
print(f"Melhor valor da função: {melhor_fitness}")
