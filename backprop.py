import numpy as np
import random
import warnings
import time

class NetworkMLP: 
    '''
        Esta classe minimiza uma função de custo utilizando o algoritmo
        Backpropagation e o otimizador Stochastic Gradient Descent com momento.
        
        Parâmetros
        ---------- 
        size_layer : list, default=[]
            Armazena o número de camadas e neorônios.
            
        learning_rate : double, default=0.01
            Taxa de aprendizado para a atualização dos pesos.
            valor entre 0 e 1
        
        tol : float, default=1e-4
            Tolerância para a otimização e fim do treinamento.
            
        momentum : float, default=0.9
            Parâmetro para atualização do gradiente descendente.
            
        max_iter : int, default=200
            Número máximo de iterações. O backpropagation é executado até
            alcançar a tolerancia ou atingir o número máximo de iterações.
        
        random_state : int, default=None
            Determina o gerador de números rândomicos para a inicialização
            de peso e viés.
            
        batch_size : int, default=1
            Tamanho dos lotes utilizados para otimização estocástica
        
        activation : {'sigmoid', 'tanh'}, default='sigmoid'
            Função de ativação para camada oculta e camada de saída
            
            'sigmoid' é a função sigmóid logística, retorna f(x) = 1 / (1 + exp(-x))
            'tanh' é a função tangente hiperbólica, retorna f(x) = a * tanh(b*x)
            
    '''
    def __init__(self, size_layer=[],
                       learning_rate=0.01,
                       momentum=0.9,
                       max_iter=200,
                       tol=1e-4,
                       random_state=None,
                       batch_size=1,
                       activation='sigmoid'):
        
        self.size_layer = size_layer       
        self.learning_rate = learning_rate
        self.tol = tol                     
        self.momentum = momentum           
        self.max_iter = max_iter           
        self.random_state = random_state
        self.batch_size = batch_size
        if activation == 'sigmoid':
            self.function = self.sigmoid
        elif activation == 'tanh':
            self.function = self.tanh
        else:
            warnings.warn(f'activation {activation} inválida.')
        
        self.biases, self.weights = self.create_param(self.size_layer, self.random_state)
        
    def create_param(self, size_layer, random_state=None):
        '''
            Função que retorna os vetores de pesos e viés gerados aleatoriamente
            na forma de distribuição Gaussiana com média 0 e variância 1. 
        '''
        np.random.seed(random_state)
        
        biases = [np.random.randn(x, 1) for x in self.size_layer[1:]]
        weights = [np.random.randn(y, x) for x, y in zip(self.size_layer[:-1], self.size_layer[1:])]
        
        return biases, weights
    
    def sigmoid(self, x, derivate=False):
        '''
            Função que retorna a ativação sigmóid ou sua derivada.
        '''
        if derivate:
            return self.sigmoid(x) * (1 - self.sigmoid(x))
        return 1./(1.+np.exp(-x))
    
    def tanh(self, x, a=1.7159, b=2/3, derivate=False):
        if derivate:
            return (b/a)*((a - self.tanh(x))*(a + self.tanh(x)))
        return a * np.tanh(b*x)
        
    def feedforward(self, z):
        '''
            Função que calcula o valor de cada neurônio da camada oculta
            e camada de saída.
        '''
        z = z.reshape(-1,1)
        dots = []            # Armazena o vetor z de cada camada 
        activations = [z]    # Armazena as ativações de cada camada
        
        for b, w in zip(self.biases, self.weights):
            z = (np.dot(w, z) + b)  # Calcula o valor dos neurônios
            dots.append(z)
            z = self.function(z)     # Aplica a função de ativação 
            activations.append(z)
            
        return dots, activations
    
    def SGD(self, training_data):
        
        n = len(training_data)
        
        batch_size = min(self.batch_size, n) # Configura o tamanho dos lotes de acordo com Nº de amostras
                
        # Dividi as amostras em lotes
        mini_batches = [training_data[k:k+batch_size] 
                        for k in range(0, n, batch_size)]
        
        # Percorre e Atualiza cada lote individualmente
        for mini_batch in mini_batches: 
            self.update_mini_batch(mini_batch) 
                
    def update_mini_batch(self, mini_batch):
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Calcula e armazena os gradientes de cada amostra pertencente a um lote
        for x, y in mini_batch: 
            gradient_bi, gradient_wi = self.backpro(x, y)
            nabla_b = [(nb*self.momentum)+(gdb) for nb, gdb in zip(nabla_b, gradient_bi)]
            nabla_w = [(nw*self.momentum)+(gdw) for nw, gdw in zip(nabla_w, gradient_wi)]
        
        # Atualização de viés e pesos
        self.biases = [b - ((self.learning_rate/len(mini_batch)) * nb) 
                       for b, nb in zip(self.biases, nabla_b)]
        
        self.weights = [w - ((self.learning_rate/len(mini_batch)) * nw) 
                       for w, nw in zip(self.weights, nabla_w)]
                
    def backpro(self, inputs, target):
        
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        
        # calcula as ativações de cada camada e a saída da rede
        dots, activations = self.feedforward(inputs) 
        
        self.y_net.append(float(activations[-1]))
        self.target.append(float(target))

        # Calcula o delta de saída e o gradiente da camada de saída
        outDelta = self.cost_derivate(activations[-1], target) * self.function(dots[-1], derivate=True)
        
        gradient_b[-1] = outDelta 
        gradient_w[-1] = np.dot(outDelta, activations[-2].T)   
        
        # Calcula o delta da oculta e seu gradiente
        z = dots[-2]
        z_d = self.function(z, derivate=True)
        hiddenDelta = np.dot(self.weights[-1].T, outDelta) * z_d # calcula o delta da oculta
        
        gradient_b[-2] = hiddenDelta
        gradient_w[-2] = np.dot(hiddenDelta, activations[-3].T)
                
        return gradient_b, gradient_w
        
    def cost_derivate(self, y_net, target):
        '''
            Retorna vetor de derivada parcial da função de custo
            em relação a saída da rede
        '''
        return (y_net - target)
    
    def fit(self, inputs, targets):
        '''
            Treina o modelo para otimização em relação os 
            dados entrada e alvos fornecidos.
        '''
        
        training_data = []
        for x, y in zip(inputs, targets):
            training_data.append((x,y))
        
        self.loss_ = []
        for time_step in range(self.max_iter):
            init = time.time()
            
            self.y_net, self.target = [], []
            
            self.SGD(training_data)
            
            squared_loss = np.mean(((np.asarray(self.y_net) - np.asarray(self.target))**2))
            
            if squared_loss <= self.tol:
                print('\nConvergência Alcançada!!!')
                break
                
            self.loss_.append(squared_loss)
                
            
            print(f'{time_step+1}-loss = {squared_loss}')
            
        end = time.time()
        print('\nRuntime : {:.4f}' .format(end-init))
        if time_step+1 == self.max_iter:
            warnings.warn('Iteração máxima alcançada, porém otimização ainda não convergiu.')