__author__ = 'FRIKIdelTO.com'
__version__ = '24.04.08'
__license__ = "GPLv3"
"""
Desarrollado a partir del siguiente cuaderno de Google Colab:
https://colab.research.google.com/drive/1lRGlbiK2wUCm07BKIhjV3dKej7jV0s1y
Todos los créditos a sus respectivos autores y colaboradores ¡¡¡Sois muy GRANDES!!!
Por defecto, se usará la GPU para sintetizar la voz si está correctamente instalada
y ésta es soportada, de lo contrario se realizará por CPU aunque es mucho más lento.
Este script dista mucho de ser perfecto así que animo a todos a realizar las
modificaciones que consideren oportunas acorde a la licencia.
Quiero agradecer también a toda la comunidad de FakeYou sin la cual esto no sería posible.
"""

# CONSTANTES
MAX_DURATION = 30
DIR_REPOSITORIOS = 'repositorios'  # donde se clonarán los repositorios
DIR_HIFIGAN = 'hifi-gan'  # carpeta del repositorio HiFi-GAN
DIR_TACOTRON2 = 'tacotron2'  # carpeta del repositorio Tacotron2
DIR_MODELOS = 'modelos'  # donde debemos guardar nuestros modelos entrenados

azul = "\33[1;36m"            # texto azul claro
azul2 = "\33[0;36m"           # texto azul oscuro
rojo = "\33[1;31m"            # texto rojo claro
amarillo = "\33[1;33m"        # texto amarillo claro
verde = "\33[1;32m"           # texto verde claro
gris = "\33[0;37m"            # texto gris
blanco = "\33[1;37m"          # texto blanco
negro_azul = "\33[0;30;106m"  # texto negro, fondo azul claro

# módulos de Python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # solo muestra errores de TensorFlow
os.makedirs(DIR_MODELOS, exist_ok=True)
import sys
# si añadimos las rutas y estas no existen, tras clonar no las encontrará
if os.path.isdir(f'{DIR_REPOSITORIOS}/{DIR_HIFIGAN}'):
    sys.path.append(f'{DIR_REPOSITORIOS}/{DIR_HIFIGAN}')
if os.path.isdir(f'{DIR_REPOSITORIOS}/{DIR_TACOTRON2}'):
    sys.path.append(f'{DIR_REPOSITORIOS}/{DIR_TACOTRON2}')
import warnings
warnings.filterwarnings("ignore")  # oculta los warnings en general
import re
import time
import json
import subprocess
from tempfile import gettempdir
from pprint import pprint
from io import BytesIO
from zipfile import ZipFile
import shutil
import stat
import threading
import wave
import traceback

def cursor_arriba(n=1):
    """
    Mueve el cursor de la terminal `n` líneas arriba.
    """
    # return
    print(f'\33[{n}A', end='')

def limpiar_pantalla():
    """
    Limpia la pantalla sin perder la información del buffer de la terminal.
    """
    # obtenemos las dimensiones de la terminal (columnas x filas)
    columnas, filas = os.get_terminal_size()
    # imprimimos tantas líneas en blanco como filas haya
    for _ in range(filas):
        print('\33[K')
    # posicionamos el cursor en la esquina superior izquierda
    print('\33[0;0H', end='')

def instalar_dependencias():
    """
    Instala las dependencias necesarias.
    """
    def instalar_modulo(modulo):
        comando = f'pip install -U {modulo}'
        if modulo == "torch":
            # añadimos la fuente para soporte CUDA
            comando+= ' --extra-index-url https://download.pytorch.org/whl/cu118'
        res = subprocess.run(comando, shell=True)
        if res.returncode != 0:
            raise Exception(f'ERROR INSTALANDO {modulo}')
        print()
    def clonar_repositorio(carpeta, url):
        def on_rm_error(func, path, exc_info):
            """
            Función auxiliar para manejar errores al eliminar archivos/directorios.
            """
            os.chmod(path, stat.S_IWRITE)
            os.unlink(path)

        print(f'{negro_azul}CLONANDO {carpeta}\33[K{gris}')
        # si git no está en el path
        if not shutil.which("git"):
            ERROR = 'ERROR: No se encuentra GIT\n'
            ERROR+= 'Puedes instalarlo desde https://git-scm.com/download\n'
            ERROR+= 'Asegúrate de añadirlo al PATH'
            print(f'{rojo}{ERROR}{gris}')
            raise Exception(ERROR)
        ruta_repo = f'{DIR_REPOSITORIOS}/{carpeta}'
        if os.path.isdir(ruta_repo):
            print(f'{amarillo}Eliminando clonación anterior{gris}')
            shutil.rmtree(ruta_repo, onerror=on_rm_error)
        res = subprocess.run(f'git clone --recursive "{url}" "{ruta_repo}"', shell=True)
        if res.returncode != 0:
            ERROR = f'ERROR CLONANDO {carpeta}'
            print(f'{rojo}{ERROR}{gris}')
            raise Exception(ERROR)
        print()

    def descargar_archivo(url, ruta_archivo):
        print(f'{negro_azul}DESCARGANDO {os.path.basename(ruta_archivo)}\33[K{gris}')
        if "drive." in url and "google." in url:
            import gdown
            gdown.download(url, ruta_archivo)
        else:
            import requests
            req = requests.get(url)
            with open(ruta_archivo, "wb") as f:
                f.write(req.content)
        print(f'Guardado en {ruta_archivo}')
        print()

    modulos = (
        'torch',
        'torchvision',
        'torchaudio',
        'torchtext',
        'torchdata',
        'num2words',
        'resampy',
        'soundfile',
        'tensorflow',
        'unidecode',
        'inflect',
        'librosa',
        'matplotlib',
        'gdown',
        'pyaudio',
        # 'requests',  # ya lo instala gdown
    )
    # instalamos los módulos
    for n, modulo in enumerate(modulos):
        print(f'{negro_azul}[{n+1} de {len(modulos)}] INSTALANDO {modulo}\33[K{gris}')
        instalar_modulo(modulo)
    # clonamos los repositorios
    clonar_repositorio(DIR_TACOTRON2, 'https://github.com/rmcpantoja/tacotron2.git')
    sys.path.append(f'{DIR_REPOSITORIOS}/{DIR_TACOTRON2}')
    clonar_repositorio(DIR_HIFIGAN, 'https://github.com/justinjohn0306/hifi-gan')
    sys.path.append(f'{DIR_REPOSITORIOS}/{DIR_HIFIGAN}')
    # descargamos HiFi-GAN Universal
    descargar_archivo(
        'https://github.com/justinjohn0306/tacotron2/releases/download/assets/g_02500000',
        f'{DIR_REPOSITORIOS}/hifigan_universal'
    )
    # descargamos HiFi-GAN Super
    descargar_archivo(
        'https://github.com/justinjohn0306/tacotron2/releases/download/assets/Superres_Twilight_33000',
        f'{DIR_REPOSITORIOS}/hifigan_super'
    )
    # descargamos el diccionario de pronunciación de Google Drive
    descargar_archivo(
        'https://drive.google.com/uc?id=1OZJ0KRjEIsIMdd21WeZaAn7CPO__8qw-',
        f'{DIR_REPOSITORIOS}/dic_pronunciacion.txt'
    )

# módulos de terceros
for _ in range(2):  # 2 intentos por si en el primer intento falta alguna dependencia
    print(f'{azul}Comprobando dependencias... {gris}')
    try:
        # módulos de terceros
        import pyaudio
        import soundfile
        import numpy as np
        import torch
        import num2words
        import resampy
        import scipy.signal
        # módulos clonados de Tacotron2
        from hparams import create_hparams
        from model import Tacotron2
        from text import text_to_sequence
        # módulos clonados de hifi-gan
        from env import AttrDict
        from meldataset import mel_spectrogram, MAX_WAV_VALUE
        from models import Generator
        from denoiser import Denoiser
        print(f'{verde}OK{gris}')
        break
    except ModuleNotFoundError:
        # print(f'{rojo}{traceback.format_exc()}{gris}')
        cursor_arriba()
        print(f'{amarillo}Instalando dependencias necesarias...{gris}')
        instalar_dependencias()

def cuadro(texto):
    """
    Imprime un cuadro con un texto dentro.
    """
    l = "─"
    c = len(texto) + 2  # +2 espacios
    print(f'{amarillo}┌{l*c}┐{gris}')
    print(f'{amarillo}│ {blanco}{texto}{amarillo} │{gris}')
    print(f'{amarillo}└{l*c}┘{gris}')

class Cronometro:

    def __init__(self):
        self.activo = False
        self.hilo = None

    def _hilo(self, texto):
        inicio = time.time()
        while self.activo:
            tiempo = time.time() - inicio
            m = int(tiempo // 60)  # minutos
            s = int(tiempo % 60)   # segundos
            time.sleep(1)
            print(f'\r{azul2}{texto} {blanco}{m}:{s:02d}{gris}\33[K ', end='', flush=True)

    def start(self, texto):
        self.activo = True
        self.hilo = threading.Thread(target=self._hilo, args=(texto,))
        self.hilo.start()

    def stop(self):
        self.activo = False
        self.hilo.join()
        print()

class FakeyouTTS:

    def __init__(self, modelo, idioma="es", device="auto"):
        """
        Args:
            modelo (str): ruta de nuestro modelo preentrenado
            idioma (str): se usa para convertir los números en letras
            device (str):
                "auto": usará GPU (cuda) si está disponible, en caso contrario CPU
                "gpu"|"cuda": fuerza el uso de GPU (cuda)
                "cpu": fuerza el uso de CPU aunque la GPU sea compatible
        """
        self.MODELO = modelo
        self.DEVICE = device
        if self.DEVICE == "gpu":
            self.DEVICE = "cuda"
        if self.DEVICE == "auto":
            print(f'{azul}Comprobando si se puede usar la GPU{gris}')
            self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            if self.DEVICE == "cuda":
                print(f'   {verde}GPU disponible{gris}')
            else:
                print(f'   {amarillo}GPU no disponible{gris}')
            print()
        print(f'{azul}Device: {blanco}{self.DEVICE.upper()}{gris}')
        Denoiser.device = self.DEVICE  # forzamos a la clase a usar el dispositivo que querramos nosotros
        self.TORCH_DEVICE = torch.device(self.DEVICE)
        self.RUTA_MODELO = f'{DIR_MODELOS}/{self.MODELO}'
        if not os.path.isfile(self.RUTA_MODELO):
            ERROR = f'ERROR: No se encuentra el modelo "{self.MODELO}"\n'
            ERROR+= f'Asegúrate de guardar tus modelos en la carpeta "{DIR_MODELOS}"'
            print(f'{rojo}{ERROR}{gris}')
            raise Exception(ERROR)
        self.IDIOMA = idioma
        self.dic_proc = {}  # diccionario básico de pronunciación
        for line in reversed((open(f'{DIR_REPOSITORIOS}/dic_pronunciacion.txt', "r").read()).splitlines()):
            self.dic_proc[(line.split(" ",1))[0]] = (line.split(" ",1))[1].strip()
        # descarga y configuración de HiFi-GAN
        self.hifigan, self.h, self.denoiser = self.conf_hifigan('universal')
        self.hifigan_sr, self.h2, self.denoiser_sr = self.conf_hifigan('super')
        # configuración de Tacotron2
        self.model, self.hparams = self.get_tacotron2()
        self.pronounciation_dictionary = False
        self.model.decoder.max_decoder_steps = MAX_DURATION * 100
        self.model.decoder.gate_threshold = 0.5
        self.superres_strength =  1.0

    def arpa(self, text, punctuation=r"!?,.;", EOS_Token=True):
        out = ''
        for word_ in text.split(" "):
            word=word_
            end_chars = ''
            while any(elem in word for elem in punctuation) and len(word) > 1:
                if word[-1] in punctuation: end_chars = word[-1] + end_chars; word = word[:-1]
                else: break
            try:
                word_arpa = self.dic_proc[word.upper()]
                word = "{" + str(word_arpa) + "}"
            except KeyError: pass
            out = (out + " " + word + end_chars).strip()
        if EOS_Token and out[-1] != ";": out += ";"
        return out

    def conf_hifigan(self, modelo):
        """
        Configura HiFi-GAN.
        """
        if modelo == "universal":
            nombre = 'hifigan_universal'
            conf = 'config_v1'
        elif modelo == "super":
            nombre = 'hifigan_super'
            conf = 'config_32k'
        # cargamos el JSON de HiFi-GAN
        with open(os.path.join(DIR_REPOSITORIOS, DIR_HIFIGAN, f'{conf}.json')) as f:
            json_config = json.loads(f.read())
        # convertimos las claves del diccionario en atributos
        try:
            h = AttrDict(json_config)
        except:
            print(f'{rojo}{traceback.format_exc()}{gris}')
            breakpoint()
        if self.DEVICE == "cpu":
            torch.manual_seed(h.seed)
        else:
            torch.cuda.manual_seed(h.seed)
        hifigan = Generator(h).to(self.TORCH_DEVICE)
        state_dict_g = torch.load(f'{DIR_REPOSITORIOS}/{nombre}', map_location=self.TORCH_DEVICE)
        hifigan.load_state_dict(state_dict_g["generator"])
        hifigan.eval()
        hifigan.remove_weight_norm()
        # eliminamos el mensaje que se genera informando de ello (soy muy tiquismiquis, ya lo sé ;-D)
        cursor_arriba()
        print('\33[K', end='', flush=True)
        denoiser = Denoiser(hifigan, mode="normal")
        return hifigan, h, denoiser

    def has_MMI(self, state_dict):
        return any(True for x in state_dict.keys() if "mi." in x)

    def get_tacotron2(self):
        """
        Carga y configura Tacotron2
        """
        hparams = create_hparams()
        hparams.ignore_layers=["embedding.weight"]
        hparams.sampling_rate = 22050
        hparams.max_decoder_steps = MAX_DURATION * 100
        hparams.gate_threshold = 0.25  # El modelo debe estar un 25% seguro de que el clip ha terminado antes de finalizar la generación
        if self.DEVICE == "cpu":
            model = Tacotron2(hparams).cpu()
        else:
            model = Tacotron2(hparams).cuda()
        state_dict = torch.load(self.RUTA_MODELO, map_location=self.TORCH_DEVICE)['state_dict']
        if self.has_MMI(state_dict):
            raise Exception("ERROR: Los modelos MMI no son compatibles con este programa")
        model.load_state_dict(state_dict)
        _ = model.eval()
        return model, hparams

    def numeros_a_palabras(self, texto):
        """
        Convierte los números del texto en letra.
        """
        texto = ' '.join([num2words.num2words(i, lang=self.IDIOMA) if i.isdigit() else i for i in texto.split()])
        return texto

    def corregir_palabras(self, texto):
        cambiar = {
            "facebook": "Feisbuc",
            "online": "onlain",
        }
        for palabra in texto.split():
            chars_strip = '?!¿¡.,:;'
            p_strip = palabra.lower().strip(chars_strip)
            if p_strip in cambiar:
                texto = re.sub(palabra.strip(chars_strip), cambiar[p_strip], texto, flags=re.IGNORECASE)
        return texto

    def tts(self, texto, archivo_salida=None):
        s = {
            "texto": texto,
            "texto_final": None,
            "archivo_salida": archivo_salida,
            "tiempo": None,
            "res": None,
        }
        inicio = time.time()
        if not s["archivo_salida"]:
            s["archivo_salida"] = f'{int(time.time())}_{self.MODELO}.wav'
        if not s["archivo_salida"].lower().endswith(".wav"):
            s["archivo_salida"]+= ".wav"

        s["texto_final"] = self.numeros_a_palabras(texto)
        s["texto_final"] = self.corregir_palabras(s["texto_final"])

        for i in [x for x in s["texto_final"].split("\n") if len(x)]:
            if not self.pronounciation_dictionary:
                if i[-1] != ";": i=i+";"
            else:
                i = self.arpa(i)
            with torch.no_grad():  # save VRAM by not including gradients
                sequence = np.array(text_to_sequence(i, ['basic_cleaners']))[None, :]
                sequence = torch.autograd.Variable(torch.from_numpy(sequence)).to(self.TORCH_DEVICE).long()
                mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(sequence)

                y_g_hat = self.hifigan(mel_outputs_postnet.float())
                audio = y_g_hat.squeeze()
                audio = audio * MAX_WAV_VALUE
                audio_denoised = self.denoiser(audio.view(1, -1), strength=35)[:, 0]

                # Resample to 32k
                audio_denoised = audio_denoised.cpu().numpy().reshape(-1)

                normalize = (MAX_WAV_VALUE / np.max(np.abs(audio_denoised))) ** 0.9
                audio_denoised = audio_denoised * normalize
                wave_audio = resampy.resample(
                    audio_denoised,
                    self.h.sampling_rate,
                    self.h2.sampling_rate,
                    filter="sinc_window",
                    window=scipy.signal.windows.hann,
                    num_zeros=8,
                )
                wave_out = wave_audio.astype(np.int16)

                # HiFi-GAN super-resolution
                wave_audio = wave_audio / MAX_WAV_VALUE
                wave_audio = torch.FloatTensor(wave_audio).to(self.TORCH_DEVICE)
                new_mel = mel_spectrogram(
                    wave_audio.unsqueeze(0),
                    self.h2.n_fft,
                    self.h2.num_mels,
                    self.h2.sampling_rate,
                    self.h2.hop_size,
                    self.h2.win_size,
                    self.h2.fmin,
                    self.h2.fmax,
                )
                y_g_hat2 = self.hifigan_sr(new_mel)
                audio2 = y_g_hat2.squeeze()
                audio2 = audio2 * MAX_WAV_VALUE
                audio2_denoised = self.denoiser(audio2.view(1, -1), strength=35)[:, 0]

                # High-pass filter, mixing and denormalizing
                audio2_denoised = audio2_denoised.cpu().numpy().reshape(-1)
                b = scipy.signal.firwin(101, cutoff=10500, fs=self.h2.sampling_rate, pass_zero=False)
                y = scipy.signal.lfilter(b, [1.0], audio2_denoised)
                y *= self.superres_strength
                y_out = y.astype(np.int16)
                y_padded = np.zeros(wave_out.shape)
                y_padded[: y_out.shape[0]] = y_out
                sr_mix = wave_out + y_padded
                sr_mix = sr_mix / normalize
                # guardamos el audio en un archivo WAV
                soundfile.write(s["archivo_salida"], sr_mix.astype(np.int16), self.h2.sampling_rate)
                print()
        # calculamos el tiempo transcurrido y devolvemos el diccionario de salida
        cursor_arriba()
        s["tiempo"] = time.time() - inicio
        s["res"] = "OK"
        return s

    def _play_wav(self, archivo_wav):
        """
        Reproduce un archivo WAV.
        """
        with wave.open(archivo_wav, 'rb') as audio:
            # instanciamos pyaudio
            api = pyaudio.PyAudio()
            # configuramos el stream
            stream = api.open(
                format = api.get_format_from_width(audio.getsampwidth()),
                channels = audio.getnchannels(),
                rate = audio.getframerate(),
                output=True
            )
            # leemos y reproducimos el archivo
            while True:
                datos_audio = audio.readframes(1)
                if not datos_audio:
                    break
                stream.write(datos_audio)
            # cerramos el stream y pyaudio
            stream.stop_stream()
            stream.close()
            api.terminate()

    def play(self, archivo_wav):
        """
        Reproduce un archivo WAV en segundo plano.
        """
        threading.Thread(target=self._play_wav, args=(archivo_wav,)).start()


# MAIN ##################################################################################################################
if __name__ == '__main__':
    limpiar_pantalla()
    cuadro(f'FakeYouTTS v{__version__} by {__author__}')
    PYTHON = os.path.splitext(os.path.basename(sys.executable))[0]
    modo_uso = f'\nModo de uso:\n'
    modo_uso+= f'    {PYTHON} {sys.argv[0]} [opciones]\n\n'
    modo_uso+= f'Opciones:\n'
    modo_uso+= f'   --install : fuerza la instalación de las dependencias\n'
    modo_uso+= f'   --cpu     : fuerza el uso de CPU aunque sea compatible con CUDA\n'
    modo_uso+= f'   --gpu     : fuerza el uso de CUDA sin comprobar si está soportado\n'
    modo_uso+= f'   --cuda    : igual que --gpu\n'
    modo_uso+= f'   --help    : muestra esta ayuda\n'
    # control de parámetros
    if "--install" in sys.argv:
        # fuerza la instalación de las dependencias
        instalar_dependencias()
    elif "--help" in sys.argv or "/?" in sys.argv:
        print(modo_uso)
        sys.exit(1)
    if "--cpu" in sys.argv:
        DEVICE = "cpu"
    elif "--gpu" in sys.argv or "--cuda" in sys.argv:
        DEVICE = "cuda"
    else:
        DEVICE = "auto"

    modelos = [x for x in os.listdir(DIR_MODELOS) if os.path.isfile(f'{DIR_MODELOS}/{x}')]
    if not modelos:
        ERROR = 'ERROR: No se encontró ningún modelo\n'
        ERROR+= f'Guarda en la carpeta "{DIR_MODELOS}" tus modelos preentrenados'
        print(f'{rojo}{ERROR}{gris}')
        sys.exit(1)
    
    # si solo hay un modelo
    n_modelos = len(modelos)
    if n_modelos == 1:
        modelo = modelos[0]
    # si hay varios
    else:
        print(f'{azul}MODELOS DISPONIBLES:{gris}')
        for i, m in enumerate(modelos):
            print(f'{blanco}{i+1}. {azul2}{m}{gris}')
        print()
        opcion = 0
        while True:
            opcion = input(f'{azul}Selecciona un modelo (1-{n_modelos}): {gris}')
            # si la opción no es un número
            if not opcion.isdigit():
                print(f'{rojo}ERROR: Debes indicar un índice de la lista{gris}')
                print()
                continue
            else:
                opcion = int(opcion)
            # si la opción está fuera del rango
            if opcion < 1 or opcion > n_modelos:
                print(f'{rojo}ERROR: Indica una opción dentro del rango (1-{n_modelos}){gris}')
                print()
                continue
            # si la opción es válida
            modelo = modelos[opcion-1]
            break


    crono = Cronometro()
    api = FakeyouTTS(modelo, device=DEVICE)

    print(f'{azul}MODELO: {blanco}{modelo}{gris}')
    print()

    while True:
        texto = input(f'{azul}Introduce un texto:{gris} ')
        if not texto:
            cursor_arriba()
            continue
        elif texto.lower() == "q":
            print(f'{amarillo}Saliendo...{gris}')
            sys.exit()
        crono.start('Generando locución:')
        res = api.tts(texto)
        crono.stop()
        if res["res"] == "OK":
            print(f'{verde}Audio guardado: {blanco}{res["archivo_salida"]}{gris}')
            api.play(res["archivo_salida"])
        else:
            print(f'{rojo}{res["res"]}{gris}')
        print()


"""
LISTADO de CAMBIOS:

24.04.06
    - Inicio del desarrollo
24.04.08
    - Añadido un cronómetro mientras se sintetiza la voz
    - Añadida reproducción el WAV en segundo plano
    - Añadidos parámetros en línea de comandos
"""


