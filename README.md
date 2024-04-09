# FakeYouTTS

Para probar modelos preentrenados de [Fakeyou](https://fakeyou.com/ "Fakeyou") (Tacotron2) en local.

Desarrollado a partir de [este cuaderno](https://colab.research.google.com/drive/1lRGlbiK2wUCm07BKIhjV3dKej7jV0s1y "este cuaderno") de Google Colab.

## Modo de uso:

`python fakeyou_tts.py [opciones]`

### Opciones

`--install` fuerza la instalación de las dependencias

`--cpu` fuerza el uso de CPU aunque sea compatible con CUDA

`--gpu` fuerza el uso de CUDA sin comprobar si está soportado

`--cuda` igual que --gpu

`--help` muestra esta ayuda

## Funcionamiento
El script instalará automáticamente las dependencias necesarias si es preciso.

Por defecto, se usará la GPU para sintetizar la voz si está correctamente instalada
y ésta es soportada, de lo contrario se realizará por CPU aunque es mucho más lento.

## Agradecimientos
Quiero agradecer a los autores y colaboradores del cuaderno de Colab y en general a toda la comunidad de FakeYou sin la cual este script no sería posible. ¡¡¡Sois muy GRANDES!!!




