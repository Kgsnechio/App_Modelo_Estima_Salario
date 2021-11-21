import streamlit as st 
import pandas as pd 
from pycaret.classification import load_model, predict_model

dados = pd.read_csv('dados_proficoes_resumo.csv')
modelo = load_model('modelo_previsao_salarios')

st.markdown('# Modelo para Estimar os Salários de Colaboradores na Área de Dados')

st.markdown('---') 
st.markdown('## Conjunto de dados *Pesquisa de proficionais na área de Dados em 2019*')
st.write(dados)
st.markdown('---') 
st.write('A média salárial para quem trabalha com dados é  **R$ {:.2f}**'.format(dados['Salário'].mean()))

st.markdown('---') 
'''
## Descrição dos dados
'''
st.table(dados['Salário'].describe())

st.markdown('---') 
campo_media = st.selectbox('Selecione uma variável para obter a média', ['Idade', 'Profissão', 'Escolaridade', 'Setor de Mercado', 'Estado'])
tabela_media = dados['Salário'].groupby(dados[campo_media]).mean()
st.write(tabela_media)
st.markdown('---') 

'''
## Quantidade de participações na pesquisa
'''
campo_qtde = st.selectbox('Selecione uma variável para obter a média', ['Idade', 'Profissão', 'Escolaridade', 'Setor de Mercado', 'Estado'])
plot = dados[campo_qtde].value_counts().plot(kind = 'barh')
st.pyplot(plot.figure)

st.markdown('---')
st.markdown('## **Modelo para Estimar o Salário de Profissionais da área de Dados**')
st.markdown('Utilize as variáveis abaixo para utilizar o modelo de previsão de salários desenvolvido abaixo.')
st.markdown('---')

col1, col2, col3 = st.columns(3)

x1 = col1.radio('Idade', dados['Idade'].unique().tolist() )
x2 = col1.radio('Profissão', dados['Profissão'].unique().tolist())
x3 = col1.radio('Tamanho da Empresa', dados['Tamanho da Empresa'].unique().tolist())
x4 = col1.radio('Cargo de Gestão', dados['Cargo de Gestão'].unique().tolist())
x5 = col3.selectbox('Experiência em DS', dados['Experiência em DS'].unique().tolist()) 
x6 = col2.radio('Tipo de Trabalho', dados['Tipo de Trabalho'].unique().tolist() )
x7 = col2.radio('Escolaridade', dados['Escolaridade'].unique().tolist())
x8 = col3.selectbox('Área de Formação', dados['Área de Formação'].unique().tolist())
x9 = col3.selectbox('Setor de Mercado', dados['Setor de Mercado'].unique().tolist())
x10 = 1
x11 = col2.radio('Estado', dados['Estado'].unique().tolist()) 
x12 = col3.radio('Linguagem Python', dados['Linguagem Python'].unique().tolist()) 
x13 = col3.radio('Linguagem R', dados['Linguagem R'].unique().tolist()) 
x14 = col3.radio('Linguagem SQL', dados['Linguagem SQL'].unique().tolist()) 
	 

dicionario  =  {'Idade': [x1],
				'Profissão': [x2],
				'Tamanho da Empresa': [x3],
				'Cargo de Gestão': [x4],
				'Experiência em DS': [x5],
				'Tipo de Trabalho': [x6],
				'Escolaridade': [x7],
				'Área de Formação': [x8],
				'Setor de Mercado': [x9],
				'Brasil': [x10],
				'Estado': [x11],		
				'Linguagem Python': [x12],
				'Linguagem R': [x13],
				'Linguagem SQL': [x14]}

dados = pd.DataFrame(dicionario)  

st.markdown('---') 
st.markdown('## **Quando terminar de preencher as informações da pessoa, clique no botão abaixo para estimar o salário de tal profissional**') 


if st.button('EXECUTAR O MODELO'):
	saida = float(predict_model(modelo, dados)['Label']) 
	st.markdown('## Salário estimado de **R$ {:.2f}**'.format(saida))

st.markdown('---') 