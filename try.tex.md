Suponer, sin pérdida de generalidad, que se tiene una red neuronal recurrente con solo tres capas: entrada, oculta y salida, y cuya capa oculta está conectada consigo misma además de con la capa de salida. Sea $n$, $m$ y $l$ el número de neuronas existentes en la capa de entrada, oculta y salida, respectivamente. Sea $t$ el indicador de los tiempos de observación. Se definen

\begin{itemize}
	\item[$\triangleright$] $x_{t}^{i}$ las entradas (\textit{inputs}) de la red en el instante $t$,  $i=1,\dots,n$
	\item[$\triangleright$] $h_{t}^{j}$ las salidas (\textit{outputs}) de la capa oculta en el instante $t$,  $j=1,\dots,m$ 
	\item[$\triangleright$] $y_{t}^{k}$ las salidas finales de la red en el instante $t$, $k=1,\dots,l$
	\item[$\triangleright$] $f$ y $g$ las funciones de salida de la capa oculta y de la capa de salida, respectivamente
	\item[$\triangleright$] $w^{1}_{ij}$ los pesos asociados a las conexiones entre las neuronas de la capa de entrada y la oculta
	\item[$\triangleright$] $w^{2}_{jh}$ los pesos asociados a las conexiones entre las las propias neuronas de la capa oculta
	\item[$\triangleright$] $w^{3}_{jk}$ los pesos asociados a las conexiones entre las neuronas de la capa oculta y la de salida
	\item[$\triangleright$] $\theta^{1}_{j}$ y $\theta^{2}_{k}$ los sesgos (\textit{bias}) asociados a la neurona $j$ de la capa oculta y la neurona $k$ de la capa de salida
\end{itemize}

Puede definirse la red neuronal recurrente mediante la siguiente expresión:

$$ h_{t}^{j}=f\bigg(\sum_{i=1}^{n}w^{1}_{ij}x_{t}^{i} + \sum_{h=1}^{m}w^{2}_{jh}h_{t-1}^{h} + \theta^{1}_{j}\bigg),\hskip0.5cm y_{t}^{k} = g \bigg( \sum_{j=1}^{l}w^{3}_{jk}h_{t}^{j} + \theta^{2}_{j}\bigg).$$
