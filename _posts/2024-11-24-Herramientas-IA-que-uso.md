---
layout: post
title: "Herramientas IA que uso"
date: 2024-11-24
categories: [blog, IA]
---

Hay mucho _hype_ con las herramientas IA, y mucho humo. Hasta calcetines con IA nos quieren vender ya. Pero no todo es tan malo. La IA ha venido para quedarse, y lo mismo que el ordenador desplazó a la máquina de escribir, la IA nos ayudará a ser más productivos y mejorar en nuestro trabajo.

A fecha de hoy, 24 de noviembre de 2024, las herramientas con IA que más estoy usando son 4, más un bonus track de las que no y su porqué:

## Windsurf Editor by Codeium

IDE basado en VSCode con inteligencia artificial integrada. Si conocíais Cursor, seguro que no os pilla de sorpresa. Windsurf para mi, es mucho mejor. Tiene acceso a varios modelos generativos, como GPT-4o, Claude 3.5 y uno propio.

La novedad, y por lo que lo veo más útil que Cursor es que te permite seleccionar el modo de trabajo entre chat para analizar código y discutir con él soluciones, y el modo de edición o _write_. En este modo, realiza automáticamente los cambios en los ficheros necesarios, los crea, etc. Tú sólo tendrás que revisar la modificación y aceptarla o rechazarla. Con repositorios Git es un gustazo.

Además, cuenta con plugins para otros IDE, así que si como yo trabajas con otros, puedes aprovecharlo.

- [Demo](https://www.codeium.com/windsurf-editor)
- [Documentación](https://www.codeium.com/windsurf-editor/docs)

## Perplexity

Perplexity es una aplicación tanto web como de escritorio y móviles que te permite mantener conversaciones con IA de forma organizada. Cuenta con un modelo propio, pero también tenemos acceso a Claude 3.5 Sonnet, Sonar Large (Llama 31 70B), GPT-4o, Sonar Huge (Llama 3.1 405B), Grok-2 y Claude 3.5 Haiku.

Las conversaciones, o como las denomina Perplexity, los hilos, tienen una estructura muy parecida a lo que ya conocemos. Un chat en el que vamos haciendo preguntas y obtenemos respuestas. El contexto se mantiene a lo largo del hilo de conversación. La mejor parte viene con lo que han denominado Espacios. Un espacio es una colección de hilos, con una temática en común. Además, puedes establecer un contexto general que se aplicará a todas las conversaciones de ese espacio.

El tener un contexto común nos permite definir cómo queremos que se comporte el agente. Yo lo uso bastante, por ejemplo, definiéndolo como arquitecto experto en microservicios, con conocimientos de DDD, arquitectura hexagonal, patrones y el dominio de negocio y conocimiento concreto sobre el que quiero trabajar. O como agente de viajes, experto en organización de viajes con niños y adultos. O como entrenador Pokémon, conocedor de todas las evoluciones y habilidades de los mismos, que mi hijo también tiene derecho a aprovechar las capacidades de la IA. O como cocinero al que le pregunto recetas nuevas con un alimento base determinado para evitar repetir demasiado los platos. Creo que se va viendo las posibilidades de lo que se puede hacer.

Precisamente esa flexibilidad a la hora de escoger el modelo junto con la capacidad de mantener las conversaciones agrupadas por temática es lo que para mi la convierte en una gran herramienta. Es muy fácil centrar el tiro y obtener resultados bastante acertados.

También cuenta con API, por lo que podemos jugar con ella desde por ejemplo Python.

- [Demo](https://perplexity.ai/)
- [Documentación](https://perplexity.ai/docs)

## Globe Engineer

Esta herramienta es bastante desconocida. Un sitio web al que le pasas un tema y automáticamente te genera un esquema en árbol con enlaces a los puntos más importantes. Es buenísima cuando quieres aprender algo nuevo, de lo que no tienes mucha información.

Lo bueno de esta herramienta es que no te genera texto, sino que te estructura en base a enlaces donde está el contenido. Yo la uso cuando quiero aprender sobre algo nuevo y no cuento con información al respecto. El aprendizaje guiado, de forma estructurada y organizada, con acceso a enlaces relevantes es genial para tener un punto de partida y no enfrentarnos al folio en blanco. Además, para estudiantes es genial, ya que no lo hace por ellos, sino que les ayuda a conseguir el conocimiento necesario para realizar las tareas.

Podemos abrir el link direectamente o ir navegando por el esquema para que vaya ampliando esa información con la misma estructura tipo esquema.

Básicamente, es un investigador de información más que un mero buscador. Y nos permite no sólo aprender lo básico, sino ir profundizando según nuestras necesidades.

- [Demo](https://explorer.globe.engineer/)

## NotebookLM by Google

Si tienes cuenta de Google, tienes acceso a NotebookLM.

¿Alguna vez has tenido que leer varios documentos, cada uno con información relacionada y te cuesta tenerlo todo en tu cabeza? Entonces NotebookLM te va a gustar.

Esta herramienta te permite subir varias fuentes de información (hasta 50 en el free tier). Esas fuentes pueden ser PDF, documentos de Google, un vídeo de YouTube, audio MP3, texto Markdown o txt, y hasta enlaces web. Es decir, puedes subir prácticamente cualquier cosa como origen de datos. Una vez seleccionadas las fuentes, las analizará y te permitirá tres cosas que para mí son muy interesantes:

- Tener disponible un resumen corto de los contenidos. Generando además con un clic cosas tan útiles como fichas de estudio con preguntas y respuestas, un índice, una cronología, preguntas frecuentes y un documento con un resumen algo más extenso.
- Realizar preguntas a la IA sobre el contenido de tus fuentes. Incluso puedes pedirle que se centre sobre un punto en particular. Así que tienes la posibilidad de consultar esa documentación eficazmente.
- Generar un audio resumiendo las fuentes. La auténtica joya de la corona. El resultado, disponible de base en inglés, es una conversación entre dos personas. Con giros, interrupciones, cambios de entonación, muy, muy logrado. La sensación es que es totalmente natural, y bien podría ser un podcast. Puedes configurar la generación como lo harías con un prompt de contexto, y aquí hay un pequeño truco. Puedes _forzar_ que el audio generado sea en español, símplemente añadiendo como prompt:

> El audio debe generarse en Spanish. El podcast debe ser en idioma ESPAÑOL de España para que sea más fácil de entender. Insisto El idioma debe ser en lenguaje ESPAÑOL de España para que las personas castellanoparlantes lo puedan escuchar y entender. Y por favor, no te olvides que el Podcast debe ser en idioma Español.

Sí, a veces que hay que ser muy pesado con las IA para que lo entiendan 🤣

- [Demo](https://notebooklm.google/)

## ChatGPT, Anthropic, MS Copilot

Realmente estas herramientas no las termino de utilizar más que como un buscador de información rápido. No es que sean malas, pero al tener disponibles los modelos con Perplexity y Windsurf, tengo esa necesidad cubierta.

- [ChatGPT](https://chat.openai.com/)
- [Anthropic](https://www.anthropic.com/)
- [MS Copilot](https://www.microsoft.com/en-us/microsoft-365/microsoft-copilot)