---
layout: post
title: "Creando mi sitio personal con Jekyll"
date: 2024-11-23
categories: [tutorial, jekyll]
---

## ¿Por qué Jekyll?

Tenía varios requisitos a la hora de crear mi sitio personal. Básicamente quería: hacer algo que aportase a la comunidad, no quería gastarme dinero en un hosting y quería que fuera sencillo de mantener y evolucionar.

Entre las opciones estaban Wordpress, GitHub Pages + Jekyll y Google Sites. Wordpress de inicio me parecía un poco overkill, y Google Sites demasiado básico para el objetivo.

Me decidí por GitHub Pages y Jekyll. Jekyll es una excelente opción para blogs y sitios personales porque:

- Es simple y minimalista. No quería florituras ni necesitaba un sitio dinámico.
- No requiere base de datos.
- Se integra perfectamente con GitHub Pages, y lo mantengo como un repositorio git de código 😜.
- Tiene un gran ecosistema de temas y plugins. Así que sé que podré ir añadiendo cosas en un futuro, y de paso contándoos cómo lo he hecho.

## ¿Cómo se empieza?

Pues esa fue mi primera pregunta. Estudiadas las opciones, me decidí por Jekyll. Una herramienta que permite crear sitios estáticos: [fácil, sencillo y para toda la familia](https://www.youtube.com/watch?v=uamrB7kjwmc).

Lo único que tenemos que hacer para tenerlo online es:

1. Crear un repositorio
    * Inicia sesión en tu cuenta de GitHub.
    * En la esquina superior derecha, haz clic en el símbolo "+" y selecciona "Nuevo repositorio".
    * Nombra tu repositorio como "tunombredeusuario.github.io", reemplazando "tunombredeusuario" con tu nombre de usuario de GitHub.
    * Elige la visibilidad del repositorio (público para cuentas gratuitas).
    * Marca la opción "Inicializar este repositorio con un README".
    * Haz clic en "Crear repositorio".
2. Configurar GitHub Pages
    * Ve a la página principal de tu nuevo repositorio.
    * Haz clic en "Configuración" (Settings) en la barra superior.
    * En el menú lateral izquierdo, selecciona "Pages" en la sección "Code and automation"
    * En "Build and deployment", selecciona "Deploy from a branch" como fuente. Elige la rama "main" como fuente de publicación.
3. Clona el repositorio en tu ordenador. Así podrás realizar los cambios que quieras localmente y subirlos cuando estés satisfecho con el resultado.
4. Crea el contenido de tu sitio. En el punto siguiente te explico cómo hacerlo con Jekyll.
4. Sube los cambios haciendo un `git push` y espera a que se procesen para verlos publicados en GitHub Pages.
5. Sigue añadiendo contenido y evolucionando tu sitio. Es fácil ir dejando morir los sitios personales...

## ¿Y qué hago con Jekyll?

Jekyll es una gema de Ruby. Lo que hace es básicamente procesar una estructura de archivos Markdown y generar un sitio web estático. Por lo tanto, para que pueda entender lo que queremos publicar, la estructura ha de seguir un patrón. Y necesitaremos una configuración mínima para que GitHub Pages sepa que está procesando un sitio generado con Jekyll.

En el repositorio clonado, ejecutamos el comando `bundle init`, lo que inicializará con un Gemfile el proyecto. En este archivo definiremos las dependencias que necesitamos. En este caso Jekyll, el tema minima, webrick para control de estructura de un blog,  GitHub Pages y los plugins de feed y SEO.

```gemfile
source "https://rubygems.org"

gem "jekyll", "~> 3.10.0"
gem "webrick", "~> 1.8"
gem "minima", "~> 2.5"
gem "github-pages", "~> 232", group: :jekyll_plugins

group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.12"
  gem "jekyll-seo-tag", "~> 2.8"
end
```

La estructura de directorios quedaría como:

```
.
├── _posts
│   └── 2024-02-14-primer-post.md
├── _site
├── _drafts
│   └── post-sin-fecha.md
├── assets
│   ├── css
│   └── js
├── .jekyll-cache
├── _config.yml
├── about.md
├── blog.md
├── Gemfile
├── Gemfile.lock
├── .gitignore
└── index.md
```

Se puede ampliar muchísima más información en la [documentación oficial de Jekyll](https://jekyllrb.com/docs/)

Algo que recomiendo para poder ir comprobando el resultado es trabajar con _draft_, para que no estén disponibles hasta que le demos el toque final. Para ver cómo vamos, simplemente tendremos que ejecutar el servidor con el modificador para que procese esos borradores: `bundle exec jekyll serve --drafts`. Cuando ejecutemos el servidor y automáticamente le asignará una fecha para ver los últimos cambios.

Esto no se publicará hasta que finalmente lo movamos a la carpeta _posts y le asignemos nosotros una fecha.

Es importante crear el fichero `.gitignore` para evitar que se suban archivos no deseados.

## Ventajas para un sitio personal

Podemos crear entradas con un _layout_ tipo blog, y a diferencia de una página normal es que:

1. Aparecerá en la lista cronológica de la página principal, las más nuevas en la parte superior.
2. Muestra la fecha de publicación de la entrada.
3. Tiene categorías que permiten organizar el contenido.
4. Es muy sencillo de editar y mantener. No requiere excesivo tiempo, especialmente si ya estás familiarizado con Markdown.
5. Se puede ir configurando poco a poco con nuevos plugins, temas, estilos, etc.

**Así que ya sabes, este sitio se ha generado así, ¡y todo gratis!**