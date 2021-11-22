import streamlit as st 
import pandas as pd 
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title = 'Estimar os Salários')

st.image('capa.jpg', use_column_width = 'always')

@st.cache
def ler_dados():
	dados = pd.read_csv('prof-dados-resumido.csv')
	dados = dados.dropna()
	return dados

dados = ler_dados()  

modelo = load_model('modelo_previsao_salarios')

st.markdown('# Modelo para Estimar os Salários de Colaboradores na Área de Dados')


st.markdown('---') 
st.markdown('## Conjunto de dados *Pesquisa de proficionais na área de Dados em 2019*')
st.markdown('Esses dados foram obtidos em *https://www.kaggle.com/*')
st.write(dados)

st.markdown('---') 
st.write('No geral a média salárial para quem trabalha com dados é de **R$ {:.2f}**'.format(dados['Salário'].mean()))

st.markdown('---') 
campo = st.selectbox('Selecione uma variável para obter a média sálarial e as quantidades de participações na pesquisa', ['Idade', 'Profissão', 'Escolaridade', 'Setor de Mercado', 'Estado'])

plot = dados['Salário'].groupby(dados[campo]).mean().plot(kind = 'barh')
st.pyplot(plot.figure)

st.markdown('---')

st.markdown('## **Modelo para Estimar o Salário de Profissionais da área de Dados**')
st.markdown('Selecione as variáveis abaixo para utilizar o modelo de previsão de salários.')
st.markdown('---')

col1, col2, col3 = st.columns(3)

x1 = col1.radio('Idade', ['18 a 24 anos', '25 a 30 anos', '31 a 40 anos', '41 a 50 anos'] )

x2 = col1.radio('Profissão', ['Analista de BI',
 							 'Analista de Dados',
							 'Cientista de Dados',
							 'Desenvolvedor/Engenheiro de Software',
							 'Engenheiro de Dados',
							 'Outras'])

x3 = col1.radio('Tamanho da Empresa', ['Pequena', 'Media', 'Grande'])

x4 = col1.radio('Cargo de Gestão', ['Sim', 'Não'])

x5 = col3.selectbox('Experiência em DS', ['Não tenho experiência na área de dados',
										'Menos de 1 ano',
										'de 1 a 2 anos',
 										'de 4 a 5 anos',
										'de 2 a 3 anos',
										'de 6 a 10 anos'
										'Mais de 10 anos']) 

x6 = col2.radio('Tipo de Trabalho', ['Estagiário',
									'Empregado (CTL)',
									'Empreendedor ou Empregado (CNPJ)',
									'Outros'] )


x7 = col2.radio('Escolaridade', ['Não tenho graduação formal',
								'Estudante de Graduação',
 								'Graduação/Bacharelado',
								'Pós-graduação',
								'Mestrado',
								'Doutorado ou Phd',
								'Prefiro não informar'] )

x8 = col3.selectbox('Área de Formação', ['Ciências Sociais',
										'Computação / Engenharia de Software / Sistemas de Informação',
										'Economia/ Administração / Contabilidade / Finanças',
										'Estatística/ Matemática / Matemática Computacional',
										'Marketing / Publicidade / Comunicação / Jornalismo',
										'Outras Engenharias',
										'Química / Física',
										'Outras'])

x9 = col3.selectbox('Setor de Mercado', ['Agronegócios',
 										'Educação',
 										'Entretenimento ou Esportes',
 										'Finanças ou Bancos',
 										'Indústria (Manufatura)',
 										'Internet/Ecommerce',
 										'Marketing',
 										'Área da Saúde',
 										'Seguros ou Previdência',
 										'Setor Alimentício',
 										'Setor Automotivo',
 										'Setor Farmaceutico',
 										'Setor Público',
 										'Tecnologia/Fábrica de Software',
 										'Telecomunicação',
 										'Varejo',
 										'Outras'])

x10 = 1

x11 = col2.radio('Estado', ['Espírito Santo (ES)',
							'Minas Gerais (MG)',
							'Paraná (PR)',
							'Rio Grande do Sul (RS)',
							'Rio de Janeiro (RJ)',
							'Santa Catarina (SC)',
							'São Paulo (SP)']) 

x12 = col3.radio('Linguagem Python', ['Sim', 'Não']) 

x13 = col3.radio('Linguagem R', ['Sim', 'Não']) 

x14 = col3.radio('Linguagem SQL', ['Sim', 'Não']) 
	
#Tratando respostas
def resposta_binaria (resp):
	if resp == 'Sim':
		return 1
	else:
		return 0

x4 = resposta_binaria (x4)
x12 = resposta_binaria (x12)
x13 = resposta_binaria (x13)
x14 = resposta_binaria (x14)


idades = {'18 a 24 anos':'[18,24]','25 a 30 anos':'[25,30]','31 a 40 anos':'[31,40]','41 a 50 anos':'[41,50]'}

x1 = idades[x1]

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

st.markdown('## **Clique no botão abaixo para estimar o salário**') 

if st.button('EXECUTAR O MODELO'):

	saida = float(predict_model(modelo, dados)['Label']) 
	st.markdown('## Salário estimado de **R$ {:.2f}**'.format(saida))

st.markdown('---') 

st.markdown('Criado por André Nechio - Disponivel em [GitHub](https://github.com/Kgsnechio/App_Modelo_Estima_Salario)')