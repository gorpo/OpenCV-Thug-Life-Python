#!/usr/bin/env python
# -*- coding: utf-8 -*-
#███╗   ███╗ █████╗ ███╗   ██╗██╗ ██████╗ ██████╗ ███╗   ███╗██╗ ██████╗
#████╗ ████║██╔══██╗████╗  ██║██║██╔════╝██╔═══██╗████╗ ████║██║██╔═══██╗
#██╔████╔██║███████║██╔██╗ ██║██║██║     ██║   ██║██╔████╔██║██║██║   ██║
#██║╚██╔╝██║██╔══██║██║╚██╗██║██║██║     ██║   ██║██║╚██╔╝██║██║██║   ██║
#██║ ╚═╝ ██║██║  ██║██║ ╚████║██║╚██████╗╚██████╔╝██║ ╚═╝ ██║██║╚██████╔╝
#╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝ ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝ ╚═════╝
#            @GorpoOrko | Manicomio TCXS Project | 2020
import cv2
from PIL import Image
import numpy as np

# classificador 
faceCascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt2.xml')
# Abrindo a imagem com PIL
imagem = Image.open('images/1.png')

def thug_mask(video):
	# convertendo a imagem pra cinza pois o cv2 reconhece melhor assim e deu!
	video_cinza = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
	# detectando a imagem em escala de cinza e botando a
	# escala para 3 pq meu pc é da xuxa, se vc tem gpu pica tira o scaleFactor
	faces = faceCascade.detectMultiScale(video_cinza, scaleFactor=3)
	# convertendo a imagem cv2 para uma imagem PIL 
	sobreposicao = Image.fromarray(video)

	for (x,y,w,h) in faces:
		# enfiando a mascara
		encaixa_imagem = imagem.resize((w, h), Image.ANTIALIAS)
		# colocando a mascara em sobreposicao
		sobreposicao.paste(encaixa_imagem, (x,y), mask=encaixa_imagem)
	# retornando o sobreposicao como uma imagem cv2
	return np.asarray(sobreposicao)


# camera de video bote zero aqui pois eu sou pobre e uso o app droidcam
# da google play se vc é pobre tambem basta por seu ip do droidcam ali e manter o final /mjpegfeed
#se vc usa notebook ou é playboy troque o valor do endereço por zero como exemplo:
#ex: cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('http://192.168.0.4:4747/mjpegfeed')

while True:
	# lendo o valor de retorno e do frame(video)
	retorno, video = cap.read()

	if retorno == True:
		# Mostrando a mascara do thuglife maconheiro
		cv2.imshow('Manicomio', thug_mask(video))

		# se apertar esc para a porra toda bro!
		if cv2.waitKey(1) == 27:
			break

# fecha a cam
cap.release()
# destroi a janela do opencv
cv2.destroyAllWindows()
