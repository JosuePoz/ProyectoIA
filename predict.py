import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import tensorflow as tf 
longitud, altura = 100, 100
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn = tf.keras.models.load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("pred: Desarrollo hojas 2")
    print("7-8 semanas de vida ")
  elif answer == 1:
    print("pred: Desarrollo hojas 1")
    print("4-6 semanas de vida ")
  elif answer == 2:
    print("pred: Germinación")
    print("1-3 semanas de vida ")
  elif answer == 3:
    print("pred: Maduración de la semilla")
    print("17-18 semanas de vida ")
  elif answer == 4:
        print("No es hoja de cilantro")
  elif answer == 5:
    print("pred: Elongación del tallo ")
    print("9-16 semanas de vida ")


  return answer

print(predict('5.jpg'))