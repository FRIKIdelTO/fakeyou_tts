Para probar modelos preentrenados de Fakeyou (Tacotron2) en local.

Modo de uso:
    python fakeyou_tts.py [opciones]

Opciones:
   --install : fuerza la instalación de las dependencias
   --cpu     : fuerza el uso de CPU aunque sea compatible con CUDA
   --gpu     : fuerza el uso de CUDA sin comprobar si está soportado
   --cuda    : igual que --gpu
   --help    : muestra esta ayuda

Desarrollado a partir del siguiente cuaderno de Google Colab:
https://colab.research.google.com/drive/1lRGlbiK2wUCm07BKIhjV3dKej7jV0s1y
Todos los créditos a sus respectivos autores y colaboradores ¡¡¡Sois muy GRANDES!!!
Por defecto, se usará la GPU para sintetizar la voz si está correctamente instalada
y ésta es soportada, de lo contrario se realizará por CPU aunque es mucho más lento.
Este script dista mucho de ser perfecto así que animo a todos a realizar las
modificaciones que consideren oportunas acorde a la licencia.
Quiero agradecer también a toda la comunidad de FakeYou sin la cual esto no sería posible.
