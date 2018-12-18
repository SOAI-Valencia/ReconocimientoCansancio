# ReconocimientoCansancio
Simple programa en python que usando tu webcam o un video detecta si tiene mucho tiempo los ojos cerrados
![demo](dormida.gif)

## Como funciona ? 
- Primero Detecta los puntos de referencia de la cara con una red neuronal ya entrenada.
- Calcula la relacion entre distancia de los ojos para detectar si estan abiertos o cerrados
- Reproduce un sonido si detecta que se ha cerrado

## Como detecta los puntos de la cara
Usa una red pre-entrenada (d'libs) que se basa en un standard Histogram of Oriented Gradients + Linear SVM method
![puntos de referencia de la cara](https://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup-768x619.jpg)

### Proporciones de ojos
![algoritmos proporcion de ojos](https://www.pyimagesearch.com/wp-content/uploads/2017/04/blink_detection_plot.jpg)


Referencia [PyImageSearch](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/)
