# Detecção de TDAH - Modelagem Preditiva
Este projeto tem como objetivo principal desenvolver um modelo de Machine Learning capaz de auxiliar na detecção de Transtorno do Déficit de Atenção com Hiperatividade (TDAH). Para isso, o pipeline inclui etapas essenciais como a coleta e pré-processamento de dados de diversas fontes médicas e psicométricas, seleção de características (features) relevantes e o treinamento de um modelo de classificação robusto, utilizando o algoritmo Random Forest.Estrutura do ProjetoA organização do repositório segue a seguinte estrutura:

tg-adhd-diagnosis/
├── adhd-diagnosis-data/
│   ├── features.csv
│   ├── CPT_II_ConnersContinuousPerformanceTest.csv
│   ├── hrv_features.csv
│   └── patient_info.csv
├── models/
│   └── modelo_RF.joblib
├── main.py  # Arquivo Python com o código principal do projeto
└── saida.txt


adhd-diagnosis-data/: Este diretório armazena os conjuntos de dados brutos, no formato CSV, que são utilizados para o treinamento e a avaliação do modelo.

models/: Contém o modelo de Machine Learning treinado, salvo no formato .joblib, pronto para ser carregado e utilizado para novas predições.

main.py: O script principal em Python que encapsula todo o pipeline: desde o carregamento e pré-processamento dos dados, a otimização de hiperparâmetros, o treinamento do modelo final, até sua avaliação e salvamento.

saida.txt: Um arquivo de log onde são registrados os resultados detalhados da avaliação do modelo, incluindo métricas de desempenho.

## Conjunto de Dados
O projeto faz uso de quatro conjuntos de dados distintos, todos localizados no diretório tg-adhd-diagnosis/adhd-diagnosis-data/:

- features.csv: Este arquivo contém um conjunto de características gerais e métricas relevantes para cada paciente participante do estudo.
- CPT_II_ConnersContinuousPerformanceTest.csv: Dados derivados do Teste de Desempenho Contínuo de Conners (CPT-II), que avalia aspectos como atenção sustentada, impulsividade e vigilância. As colunas brutas do teste (Trial e Response) são removidas no pré-processamento para focar em métricas resumidas.
- hrv_features.csv: Inclui características relacionadas à Variabilidade da Frequência Cardíaca (HRV), que podem ser indicativos do estado do sistema nervoso autônomo.
- patient_info.csv: Contém informações demográficas (como idade e gênero) e dados clínicos dos pacientes, incluindo o crucial rótulo de diagnóstico de TDAH (ADHD). Este arquivo é filtrado para incluir apenas pacientes válidos (filter_$ == 1).

## Pré-processamento de Dados
A função get_train_test_split é a espinha dorsal do processo de preparação dos dados, realizando as seguintes operações:
1. Carregamento de Dados: Lê os arquivos CSV especificados e os transforma em DataFrames do Pandas.
2. Tratamento de Valores Nulos e Colunas Vazias: Identifica e remove colunas que contêm apenas valores nulos. Para os valores ausentes restantes, aplica estratégias de preenchimento:
   - Colunas de tempo (ACC_TIME, HRV_TIME): Convertidas de HH:MM:SS para segundos e preenchidas com a moda.
   - Variáveis numéricas (ACC_DAYS, HRV_HOURS, WURS, ASRS, MADRS, HADS_A, HADS_D): Preenchidas com a mediana de suas respectivas colunas.
   - Variáveis de medicação (MED_Antidepr, MED_Moodstab, etc.): Valores nulos são preenchidos com 0, e a coluna é convertida para tipo inteiro, transformando NaN e 1 em 0 e 1 respectivamente.Filtragem de Dados: Garante que apenas os pacientes com filter_$ == 1 sejam considerados, assegurando a relevância e qualidade dos dados.Fusão de DataFrames: Os DataFrames são combinados com base no ID do paciente para criar um conjunto de dados unificado. Se patient_info não for explicitamente mesclado, suas colunas filter_$, ADHD, ADD e ID são adicionadas posteriormente para garantir a disponibilidade do alvo e do filtro.Divisão em Conjuntos de Treino e Teste: O conjunto de dados final é dividido em X_train, X_test, y_train e y_test, com 20% dos dados reservados para teste (test_size=0.2). A coluna ADHD é definida como a variável alvo (y).Balanceamento de Classes: Para mitigar o problema de desbalanceamento de classes (comum em conjuntos de dados de diagnóstico), a técnica SMOTE (Synthetic Minority Over-sampling Technique) é aplicada ao conjunto de treinamento, gerando amostras sintéticas da classe minoritária.Modelagem e TreinamentoO modelo central deste projeto é um RandomForestClassifier, conhecido por sua robustez e capacidade de lidar com dados complexos. O processo de modelagem é dividido em duas etapas principais:Otimização de Hiperparâmetros e Seleção Inicial de Features (get_best_model_features):Um GridSearchCV é empregado para realizar uma busca exaustiva pelos melhores hiperparâmetros para o RandomForestClassifier. Os parâmetros otimizados incluem n_estimators, max_depth, min_samples_split e min_samples_leaf, utilizando validação cruzada com cv=10.Após o treinamento com os melhores parâmetros e todas as features disponíveis, o modelo é avaliado. As métricas accuracy e roc_auc_score são calculadas, e um classification_report detalhado é gerado.A importância de cada feature é extraída do modelo treinado e ordenada, servindo como base para a seleção das features mais relevantes.Treinamento do Modelo Final com Features Selecionadas (get_RF_model):Com base na importância das features calculada na etapa anterior, um número predefinido de features (por padrão, as 30 mais importantes) é selecionado.Um novo RandomForestClassifier é instanciado com os hiperparâmetros otimizados e treinado exclusivamente com as features selecionadas.O desempenho deste modelo final é rigorosamente avaliado usando accuracy, roc_auc_score, confusion_matrix e classification_report. Todos esses resultados são gravados no arquivo saida.txt.Finalmente, o modelo treinado é salvo no diretório models/ como modelo_RF.joblib, permitindo sua fácil reutilização.Como ExecutarPara configurar e executar o projeto em seu ambiente local, siga os passos abaixo:Clone o Repositório:git clone <URL_DO_SEU_REPOSITORIO>
cd tg-adhd-diagnosis
Substitua <URL_DO_SEU_REPOSITORIO> pelo link real do seu repositório.Crie e Ative um Ambiente Virtual (Altamente Recomendado):python -m venv venv
# Para Linux/macOS
source venv/bin/activate
# Para Windows
venv\Scripts\activate
Instale as Dependências:pip install pandas numpy scikit-learn imbalanced-learn joblib
Execute o Script Principal:python main.py
Após a conclusão da execução, o modelo treinado estará disponível em tg-adhd-diagnosis/models/modelo_RF.joblib, e os resultados detalhados da avaliação poderão ser consultados em tg-adhd-diagnosis/saida.txt.ResultadosOs resultados da avaliação do modelo, incluindo acurácia, AUC-ROC, matriz de confusão e o relatório de classificação completo, são sistematicamente registrados no arquivo saida.txt. Além disso, as features consideradas mais importantes pelo modelo e seus respectivos pesos são extraídos e também podem ser analisados para insights adicionais sobre os fatores que mais contribuem para a detecção de TDAH.PersonalizaçãoO pipeline permite flexibilidade na seleção dos dados e no número de features utilizadas:Seleção de DataFrames: A lista dfs_to_merge no main.py pode ser modificada para incluir ou excluir conjuntos de dados na fusão. Basta descomentar ou comentar as linhas correspondentes:dfs_to_merge = [
    'ft_df',
    # 'patient_info', # Descomente para incluir o DataFrame de informações do paciente
    'cpt_2_filtered',
    # 'hrv_df'        # Descomente para incluir o DataFrame de features de HRV
]
Número de Features Finais: O parâmetro number_of_features na chamada da função get_RF_model permite controlar quantas das features mais importantes serão usadas para treinar o modelo final. Por exemplo, para usar as 50 features mais importantes:final_model_features = get_RF_model(tts, best_model_features, 50) # Altere 30 para o número desejado de features
