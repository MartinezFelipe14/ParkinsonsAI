------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Este projeto utiliza um modelo de IA para calcular a probabilidade de uma pessoa ter Parkinson, com base em um conjunto de dados disponibilizado pelo UC Irvine Machine Learning Repository. O modelo foi inspirado nas pesquisas conduzidas por Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig e outros, conforme documentado nos seguintes artigos:

- 'Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection', Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM. BioMedical Engineering OnLine 2007, 6:23 (26 June 2007)
- A Tsanas, MA Little, PE McSharry, LO Ramig (2009) 'Accurate telemonitoring of Parkinson.s disease progression by non-invasive speech tests', IEEE Transactions on Biomedical Engineering.

Esses estudos foram fundamentais para o desenvolvimento do modelo de IA, que utiliza medições de voz para identificar características associadas ao Parkinson. O objetivo deste projeto é fornecer uma ferramenta útil para ajudar na detecção precoce e no monitoramento da progressão da doença.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

name: identificador para cada registro, como o nome ou um código identificador.
MDVP:Fo(Hz): Frequência fundamental, medida em Hertz (Hz), que é a frequência mais baixa de um som complexo, geralmente associada à voz.
MDVP:Fhi(Hz): Frequência máxima medida em Hertz (Hz).
MDVP:Flo(Hz): Frequência mínima medida em Hertz (Hz).
MDVP:Jitter(%): Uma medida de flutuação na frequência fundamental da voz, geralmente relacionada à instabilidade na produção vocal.
MDVP:Jitter(Abs): Uma medida absoluta de flutuação na frequência fundamental.
MDVP:RAP: Uma medida do ciclo de flutuação na frequência fundamental, similar ao Jitter.
MDVP:PPQ: Outra medida de flutuação na frequência fundamental.
Jitter:DDP: Uma medida derivada do Jitter, geralmente é o dobro do valor do Jitter RAP.
MDVP:Shimmer: Uma medida da variação na amplitude da onda sonora da voz.
MDVP:Shimmer(dB): Uma medida em decibéis da variação na amplitude da onda sonora.
Shimmer:APQ3: Uma medida de variação na amplitude da onda sonora usando o valor dos primeiros três quartis.
Shimmer:APQ5: Uma medida de variação na amplitude da onda sonora usando o valor dos primeiros cinco quartis.
MDVP:APQ: Outra medida de variação na amplitude da onda sonora.
Shimmer:DDA: Uma medida derivada do Shimmer, calculada pela diferença entre os valores dos quartis.
NHR: Razão de ruído harmônico, uma medida de ruído na voz em relação ao sinal harmônico.
HNR: Relação sinal-ruído harmônico, uma medida global de qualidade do sinal vocal.
status: Uma variável que indica o status do paciente, que pode ser relevante para o diagnóstico ou classificação.
RPDE: Dimensão de entropia de recorrência, uma medida da complexidade da voz.
DFA: Análise fractal de flutuação, outra medida de complexidade da voz.
spread1: Um parâmetro acústico derivado da análise de distúrbios vocais.
spread2: Outro parâmetro acústico derivado da análise de distúrbios vocais.
D2: Dimensão fractal, uma medida da complexidade da voz baseada em geometria fractal.
PPE: Entropia de perturbação de pitch, uma medida da variabilidade na frequência fundamental da voz.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
