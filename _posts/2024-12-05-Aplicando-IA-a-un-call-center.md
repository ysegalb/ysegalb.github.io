---
layout: post
title: "Aplicando IA a un call center"
date: 2024-12-05
categories: [tutorial, IA]
---

## ¿Por qué esto no son otros calcetines con IA?

Hoy en día, hemos asistido a la explosión de la IA y, en particular, de los LLM generativos como ChatGPT. Es difícil encontrar un producto que, habiendo sido actualizado o lanzado en el último año, no lleve alguna referencia a la IA. Además, habiendo vivido otras revoluciones similares, sabemos que en muchas ocasiones se fuerzan los casos de uso para decir que están hechos con Inteligencia Artificial. Esto ha generado una saturación en el mercado e, incluso, a veces, cierto recelo cuando vemos las consabidas siglas.

¿Quiere decir eso que su aplicación no es útil? Para nada. Hay casos que, si bien pueden no ser novedosos en cuanto al planteamiento, se han convertido en soluciones mucho más asequibles, tanto para desarrollar como para mantener. Y uno de esos casos es el que vamos a presentar en este tutorial.

Si alguna vez habéis tenido relación con un centro de atención al cliente, sabréis que todas las conversaciones son almacenadas, ya sea en forma de audios o como chats con los agentes. Esto es necesario para aportar trazabilidad a los procesos, gestionar reclamaciones, etc. Por lo tanto, es algo que ya tenemos de base: una gran cantidad de información por explotar. Lo que propongo con este tutorial es aplicar los LLM al análisis de esos datos para mejorar los procesos de las compañías y, por supuesto, la atención al cliente.

Veremos cómo, de una manera muy sencilla, podemos procesar las transcripciones de esas conversaciones para obtener datos valiosos que ayuden a mejorar: resúmenes de las conversaciones, categorización, detección del tono de la conversación e, incluso, una lista de palabras clave.

## ¿Por dónde empezamos?

El proyecto está organizado siguiendo una estructura muy sencilla, ya que es únicamente una prueba de concepto. Para su uso en producción, requeriría trabajo adicional, que comentaremos en los últimos puntos, incluyendo posibles mejoras y aspectos que podemos implementar para acercarnos más al mundo de la IA.

### Estructura del proyecto

La estructura del proyecto es la siguiente:

```
customer-care-llm/
├── app/
│   ├── api/
│   ├── models/
│   ├── services/
│   ├── training/
│   ├── configuration.py
│   └── main.py
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

Esta estructura permite separar los distintos módulos según su responsabilidad, mejorando la escalabilidad y mantenibilidad. Los tests necesarios están incluidos dentro de cada carpeta. A continuación, describiremos brevemente el contenido de cada una:

* _app_ aplicación general. Todo estará contenido dentro de esta carpeta.
* _api_ endpoints de nuestra aplicación; será nuestro punto de interacción con el exterior.
* _models_ representación del modelo; define la entrada y salida que usará el API.
* _services_ integración con los distintos modelos mediante la instanciación de servicios y métodos para el análisis realizado por los LLM definidos.
* _training_ dado que algunos modelos requieren un pequeño entrenamiento, aquí se encuentran las factorías que nos devuelven un modelo entrenado según nuestras necesidades, así como los datos de entrenamiento necesarios.
* _.env_ definición de variables para la configuración de la aplicación, como los modelos a instanciar, datos de contacto, etc.
* _requirements.txt_ gestión de dependencias necesarias para ejecutar nuestra aplicación.

Además el proyecto incluye un README.md con las instrucciones para ejecutar.

### Modelos utilizados

En este proyecto, he utilizado distintos modelos capaces de procesar el lenguaje natural (NLP) para analizar varios aspectos de las conversaciones.
Estos modelos se han seleccionado por su buen rendimiento en tareas de NLP en español y su capacidad para procesar el lenguaje en contexto. Todos los modelos están disponibles de forma gratuita en [Hugging Face](https://huggingface.co/).

#### Análisis de sentimientos

Utilizo el modelo `nmarinnn/bert-bregman` para determinar el tono emocional de la conversación. El resultado será un valor que podemos interpretar como negativo, positivo o neutro.

#### Clasificación de texto

Empleo un modelo personalizado basado en `bert-base-spanish-wwm-cased` para categorizar el tipo de incidencia. El entrenamiento del modelo asocia frases de ejemplo con las etiquetas que queremos gestionar, permitiéndonos así establecer las categorías que necesitemos.

#### Generación de resúmenes

Utilizo el modelo `mrm8488/bert-spanish-cased-finetuned-summarization` para crear resúmenes concisos de las conversaciones. Es importante tener cuidado si el tamaño del resumen supera al del texto original, ya que puede empezar a alucinar, generando contenido adicional hasta alcanzar el tamaño deseado.

#### Reconocimiento de entidades nombradas (NER)

Las entidades nombradas son simplemente palabras clave. Para obtener un resultado coherente y significativo, además de aplicar el modelo `dccuchile/bert-base-spanish-wwm-cased` para extraer palabras clave relevantes, realizo un postproceso con _SpaCy_ para filtrar únicamente sustantivos, adjetivos y adverbios. De esta forma, evitamos que aparezcan palabras como artículos, preposiciones, etc., que no aportan significado.

## Al turrón: analicemos el código

### Configuración general
<!-- file: app/configuration.py -->
```python

import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
import torch

load_dotenv()

class Settings(BaseSettings):
    app_name: str = os.getenv("APP_NAME", "CallCenterAnalysisApp")
    admin_email: str = os.getenv("ADMIN_EMAIL", "admin@example.com")
    items_per_user: int = int(os.getenv("ITEMS_PER_USER", "50"))
    sentiment_model_path: str = os.getenv("SENTIMENT_MODEL_PATH", "nmarinnn/bert-bregman")
    classification_model_path: str = os.getenv("CLASSIFICATION_MODEL_PATH", "bert-base-spanish-wwm-cased")
    summarization_model_path: str = os.getenv("SUMMARIZATION_MODEL_PATH", "mrm8488/bert-spanish-cased-finetuned-summarization")
    ner_model_path: str = os.getenv("NER_MODEL_PATH", "dccuchile/bert-base-spanish-wwm-cased")

    class Config:
        env_file = ".env"

settings = Settings()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATHS = {
    "sentiment": settings.sentiment_model_path,
    "classification": settings.classification_model_path,
    "summarization": settings.summarization_model_path,
    "ner": settings.ner_model_path
}
```

Aquí definimos las configuraciones globales, como los modelos a utilizar, el uso de CUDA o CPU para la ejecución, etc. De esta forma, ganamos flexibilidad y mantenemos un único punto de gestión para los datos variables. Esto se puede mejorar añadiendo parámetros adicionales para la configuración de los LLM. Todos los valores se pueden especificar como variables de entorno o definir en un archivo .env, tal como se indica en el README del proyecto.

### Servicios de análisis

<!-- file: app/services/sentiment_analysis.py -->
```python
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from configuration import DEVICE, settings

class SentimentAnalysisService:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(settings.sentiment_model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(settings.sentiment_model_path).to(DEVICE)
        self.pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=DEVICE)

    def analyze(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()

        class_labels = {0: "negativo", 1: "neutro", 2: "positivo"}
        return class_labels[predicted_class]
```

El modelo está preentrenado, por lo que solo necesitamos enviarle el texto para que lo analice y luego traducir su respuesta a un formato legible.

<!-- file: app/services/text_classification.py -->
```python
import torch
from transformers import pipeline, AutoTokenizer

from configuration import DEVICE, settings, TEXT_CLASSIFICATION_CATEGORIES
from training.text_classification_factory import train_model

class TextClassificationService:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(settings.text_classification_model_path)
        self.model = train_model()
        self.pipeline = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=DEVICE)

    def classify(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        return TEXT_CLASSIFICATION_CATEGORIES[predicted_class]
```

A diferencia del servicio anterior, aquí necesitamos realizar un pequeño entrenamiento del modelo para que nos devuelva etiquetas significativas para nuestro caso específico. Esto requiere indicarle cómo relacionar tipos de frases con un conjunto de etiquetas definido. Esta parte es especialmente potente, ya que nos permite categorizar las conversaciones de una forma no preestablecida por el LLM.

<!-- file: app/services/extractive_summary.py -->
```python
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

from configuration import DEVICE, settings

class ExtractiveSummaryService:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(settings.extractive_summary_model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(settings.extractive_summary_model_path).to(DEVICE)
        self.pipeline = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=DEVICE)

    def summarize(self, text):
        result = self.pipeline(text, max_length=100, min_length=30, do_sample=False)
        return result[0]['summary_text']
```

Quizá el más sencillo de todos. Únicamente tenemos que enviarle el texto y las longitudes que queremos como límite.

<!-- file: app/services/named_entity_recognition.py -->
```python
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from configuration import DEVICE, settings
from collections import Counter

class NamedEntityRecognitionService:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(settings.ner_model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(settings.ner_model_path).to(DEVICE)
        self.pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device=DEVICE)
        self.nlp = spacy.load("es_core_news_sm")

    def extract_keywords(self, text):
        results = self.pipeline(text)

        # Filtrar y limpiar las entidades reconocidas
        entities = [item['word'].strip('#').lower() for item in results if item['score'] > 0.5 and len(item['word']) > 1]

        # Análisis morfológico para restringir a sustantivos, adjetivos y verbos
        doc = self.nlp(text)
        valid_pos = {'NOUN', 'ADJ', 'VERB'}
        filtered_entities = [
            token.text.lower() for token in doc
            if token.pos_ in valid_pos and token.text.lower() not in entities
        ]

        # Contar las ocurrencias de cada entidad
        entity_counts = Counter(filtered_entities)

        # Seleccionar las 5 entidades más frecuentes como palabras clave
        keywords = [word for word, _ in entity_counts.most_common(5)]
        return keywords
```

Es el único servicio que contiene algo de lógica de procesamiento fuera de los LLM. En este caso, el LLM identificará palabras con una determinada importancia (proceso NER), y podremos filtrarlas según esa puntuación. Para evitar que aparezcan términos sin significado concreto, nos apoyamos en SpaCy para procesar esa lista y quedarnos únicamente con las palabras que cumplan el criterio de ser sustantivos, adjetivos o adverbios mediante el análisis morfológico. Este filtro es también configurable y, como se puede ver, bastante sencillo de implementar. Por lo tanto, solo aquellas palabras que se encuentren en ambas listas serán aptas para considerarse palabras clave de la conversación.


<!-- file: app/api/endpoints.py -->
```python
from fastapi import APIRouter, HTTPException
from models.schemas import CallRequest, CallAnalysis
from services.sentiment_analysis import SentimentAnalysisService
from services.text_classification import TextClassificationService
from services.extractive_summary import ExtractiveSummaryService
from services.named_entity_recognition import NamedEntityRecognitionService

router = APIRouter()

# Instancias de servicios
sentiment_service = SentimentAnalysisService()
classification_service = TextClassificationService()
summarization_service = ExtractiveSummaryService()
ner_service = NamedEntityRecognitionService()

@router.post("/analyze_call", response_model=CallAnalysis)
async def analyze_call(call: CallRequest) -> CallAnalysis:
    try:
        # Procesar la transcripción
        transcription = call.transcription
        sentiment = sentiment_service.analyze(transcription)
        category = classification_service.classify(transcription)
        summary = summarization_service.summarize(transcription)
        keywords = ner_service.extract_keywords(transcription)

        # Retornar el análisis de la llamada
        return CallAnalysis(
            transcription=call.transcription,
            sentiment=sentiment,
            category=category,
            summary=summary,
            keywords=keywords
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

Aquí estamos definiendo una API muy sencilla, con un único endpoint POST en el que podemos proporcionar la transcripción de la conversación y obtener como respuesta su análisis. A continuación, se muestra un pequeño ejemplo de una petición y su respuesta:

<!-- file: request.json -->
```json
{
  "transcription": "Buenas tardes, querría información sobre los productos de inversión que puedo contratar, dado mi historial crediticio y de solvencia. Estaría interesado en productos de bajo riesgo, con una rentabilidad moderada y que permitan aportaciones periódicas. ¿Tienen algún tipo de producto que haga uso del interés compuesto? ¿Me pueden asegurar si los sectores en los que invierten no incluyen armamento ni actividades moralmente cuestionables? Gracias."
}
```

<!-- file: response.json -->
```json
{
  "transcription": "Buenas tardes, querría información sobre los productos de inversión que puedo contratar, dado mi historial crediticio y de solvencia. Estaría interesado en productos de bajo riesgo, con una rentabilidad moderada y que permitan aportaciones periódicas. ¿Tienen algún tipo de producto que haga uso del interés compuesto? ¿Me pueden asegurar si los sectores en los que invierten no incluyen armamento ni actividades moralmente cuestionables? Gracias.",
  "sentiment": "positivo",
  "category": "Solicitud de Servicio",
  "summary": "Buenas tardes, querría información sobre los productos de inversión que puedo contratar dado mi historial crediticio y de solvencia. Estaría interesado en productos de bajo riesgo, con una rentabilidad moderada y que permitan aportaciones periódicas. ¿Tienen algún tipo de producto que haga uso del interés compuesto? ¿Me pueden asegurar si los sectores",
  "keywords": [
    "crediticio",
    "solvencia",
    "inversión",
    "cuestionables"
  ]
}
```

Podemos observar que, al tratarse de un texto corto y de un modelo preentrenado sin ajustes específicos, el resumen no es de gran calidad. Por supuesto, existen alternativas de pago mucho más fiables y, dedicando tiempo y cuidado, también es posible obtener buenos resultados con modelos gratuitos.

## Conclusiones

Como podemos ver, no todo es humo en el mundo de la IA. Hay aplicaciones que, de forma tradicional, nos costarían muchísimo más, tanto en trabajo como en mantenibilidad. Es importante saber identificar los casos de uso donde estas soluciones aportan valor y aprovecharlos sin dudar.

Aunque este proyecto no es apto para un sistema real, sí puede servirnos como punto de partida para plantearnos lo que podríamos llegar a conseguir integrando este tipo de soluciones en el portafolio de nuestra compañía. Los próximos pasos a seguir nos pueden dar una idea de hasta qué punto podemos mejorar esta solución y aproximarnos a un sistema que aproveche la potencia que los LLM ofrecen para optimizar tanto nuestros procesos como la relación que mantenemos con los clientes.


### Posibles próximos pasos

* Implementar un sistema de logging para rastrear el uso de la API, monitorear el rendimiento y detectar errores.
* Añadir autenticación y autorización: todas nuestras aplicaciones deben estar correctamente protegidas frente a un uso fraudulento.
* Integración con aplicaciones CRM existentes: si como parte del flujo alimentamos el sistema, podremos tener en tiempo real (NRT) un análisis de las conversaciones a disposición de los agentes.
* Implementar retroalimentación para permitir la corrección o mejora de las predicciones del modelo.
* Expandir las capacidades de análisis: mejorar la detección de emociones específicas, identificación de problemas recurrentes, etc.
* Optimización del rendimiento: existen modelos más ligeros que pueden reducir la latencia en las respuestas, técnicas de inferencia más rápida, uso de otros modelos, etc.
* Implementar pruebas de carga para evaluar el rendimiento real del sistema y su capacidad de análisis.
* Explotación de los resultados: almacenar las respuestas en una base de datos para el análisis de patrones, creación de cuadros de mando, etc.

Hay multitud de posibilidades para hacer crecer el proyecto, y seguro que si le preguntamos a ChatGPT, nos puede sugerir alguna más 😉.

## Glosario de términos

🤖_generated by IA_
* LLM (Large Language Model): Un tipo de modelo de inteligencia artificial especializado en el procesamiento de lenguaje natural y la generación de texto.
* Trazabilidad: Capacidad de rastrear y verificar el historial, uso o localización de un producto o información en un proceso.
* EndPoints: Puntos de acceso en una API (Interfaz de Programación de Aplicaciones) que permiten la comunicación entre diferentes sistemas.
* Pipeline: En el contexto de aprendizaje automático, una serie de pasos o transformaciones aplicadas a los datos para entrenar o utilizar un modelo.
* CUDA: Una plataforma de computación paralela desarrollada por NVIDIA, que permite utilizar las GPU para procesamiento en lugar de la CPU.
* API Router: Un sistema de rutas en una API que organiza y redirige solicitudes a diferentes endpoints según la funcionalidad requerida.
* Modelo BERT: Un tipo de modelo de aprendizaje automático para procesamiento de lenguaje natural que permite entender el contexto completo de una palabra en una oración.
* Pydantic: Biblioteca de validación de datos para Python que facilita la creación de modelos de datos basados en clases.
* SpaCy: Herramienta avanzada de procesamiento de lenguaje natural (NLP) en Python, utilizada para análisis léxico y morfológico del texto.
* NER (Named Entity Recognition): Técnica de procesamiento de lenguaje natural que permite identificar nombres de personas, organizaciones, lugares, etc., en un texto.
* Postprocesamiento: Paso en el procesamiento de datos que se realiza después de la ejecución de un modelo, para mejorar o ajustar los resultados obtenidos.
* Softmax: Función matemática utilizada en modelos de aprendizaje automático para normalizar valores en probabilidades.
* Inferencia: En el contexto de IA, proceso de aplicar un modelo entrenado para hacer predicciones o clasificaciones en datos nuevos.
* Retroalimentación (Feedback): Proceso de ajuste o mejora continua mediante el análisis de resultados o respuestas, especialmente importante en el entrenamiento de modelos de IA.
* NRT (Near Real-Time): Procesamiento casi en tiempo real, permitiendo obtener datos o resultados poco después de la entrada de datos.

## Referencias

* [Código del tutorial en GitLab](https://gitlab.com/tutoriales-ysegura/customer-care-llm)
* [Hugging Face](https://huggingface.co/)