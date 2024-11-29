---
layout: post
title: "Informes dinámicos con Dynamic Jasper"
date: 2024-11-29
categories: [tutorial, JasperReports]
---

##### **DISCLAIMER:** Este tutorial es una versión actualizada del que realizé en 2011 y disponible en [adictos al trabajo](https://adictosaltrabajo.com/2011/02/22/dynamic-jasper/). No obstante, y aunque actualizado a la última versión de DynamicJasper, cabe destacar que la versión 5.3.9 disponible en Maven Central es del agosto del 2023. En esa fecha se cerró la última issue del proyecto. Por lo tanto, se recomienda encarecidamente hacer un fork del repositorio de GitHub para poder evolucionar la librería en caso de ser descontinuada por los desarrolladores originales o corregir posibles errores detectados.

---

Hoy en día es muy común que por pequeña que sea nuestra aplicación tengamos que generar algún tipo de informe. En el mundo Java existe un estándar de facto llamado JasperReports, ampliamente extendido. Si bien generar un informe al vuelo con JasperReports es técnicamente posible, requiere de un exhaustivo conocimiento de la herramienta, su comportamiento y posibilidades.
Para simplificar esa tarea y ayudarnos en la medida de lo posible, disponemos de un framework llamado DynamicJasper, que apoyándose en JasperReports, oculta su complejidad y nos permite realizar informes profesionales de una manera muy sencilla.

## Creación del proyecto con Maven

Lo primero que necesitamos es importar la librería de DynamicJasper. Optamos por la manera más sencilla, que es crearnos un proyecto Maven, en el que incluimos el repositorio de DynamicJasper y su dependencia:

```xml
...
  <repositories>
    <repository>
      <id>fdvsolution.public</id>
      <url>http://nexus.fdvs.com.ar/content/groups/public/</url>
    </repository>
  </repositories>
  ...
  <dependencies>
    ...
    <dependency>
      <groupId>ar.com.fdvs</groupId>
      <artifactId>DynamicJasper</artifactId>
      <version>5.3.9</version>
    </dependency>
    ...
  </dependencies>
...
```

Con esta sencilla configuración ya estamos listos para utilizar DynamicJasper en nuestro proyecto.

## Nuestro primer informe

La estructura de nuestro primer informe va a ser muy sencilla. Va a constar de un encabezado de página, un subtítulo que indique la fecha y hora de generación y una linea de detalle en la que mostrar nuestra información.

Lo que a priori parece sencillo, utilizando JasperReports directamente supondría realizar tareas de configuración del informe, definición de elementos, gestión de bandas de impresión, parámetros, etc. Ahora veremos como queda resuelto con este framework.

Para el caso que nos ocupa, vamos a dividir nuestro proyecto en datasource, exporter y clase principal. Únicamente vamos a comentar las clases de generación y exportación del informe. Para el resto del proyecto, podéis importarlo desde el [repositorio de GitHub](https://github.com/ysegura/TutorialDynamicJasper.git).

```java
public DynamicReport buildReport() throws ClassNotFoundException {
        FastReportBuilder fastReportBuilder = new FastReportBuilder();

        fastReportBuilder.addColumn("ID", "id", Long.class.getName(), 50)
                .addColumn("Nombre", "firstname", String.class.getName(), 150)
                .addColumn("Apellidos", "surname", String.class.getName(), 150)
                .addColumn("Fecha\\nIncorporación", "startDate", String.class.getName(), 160)
                .addColumn("Salario", "salary", String.class.getName(), 120)
                .addColumn("Departamento", "department", String.class.getName(), 240)
                .setTitle("Mi primer Informe con DynamicJasper").setSubtitle("Generado el " + new Date())
                .setPrintBackgroundOnOddRows(true)
                .setUseFullPageWidth(true);

        return fastReportBuilder.build();
    }
```

```java
public static void exportReport(JasperPrint jp, String path) throws JRException, FileNotFoundException {
        LOGGER.info("Exporting report to: " + path);
        JRPdfExporter exporter = new JRPdfExporter();

        File outputFile = new File(path);
        File parentFile = outputFile.getParentFile();
        if (parentFile != null) parentFile.mkdirs();
        FileOutputStream fos = new FileOutputStream(outputFile);

        exporter.setExporterInput(new SimpleExporterInput(jp));
        exporter.setExporterOutput(new SimpleOutputStreamExporterOutput(fos));

        exporter.exportReport();

        LOGGER.info("Report exported: " + path);
    }
```

Como podemos ver, la generación de informes sencillos de manera dinámica no podría ser más simple. Sólo tenemos que instanciar el builder que nos proporciona DynamicJasper, ir añadiendo las columnas que necesitemos en el orden de salida, títulos y demás propiedades. La exportación se delega a los métodos de JasperReports, ya que lo que obtenemos es un objeto JasperPrint.

Y por si quieres probarlo tú, te dejo un ejemplo de uso en el [repositorio de GitHub](https://github.com/ysegalb/tutorial-dynamic-jasper).

## Conclusiones

Si bien hemos conseguido generar un informe muy simple, DynamicJasper nos brinda multitud de posibilidades a la hora de trabajar con reports. Podemos gestionar virtualización, grupos, layouts, tablas, estilos, subreports, plantillas, etc. con la misma facilidad que este pequeño informe. Merece la pena dedicar un tiempo a revisar todas las posibilidades que nos ofrece esta librería.

## Referencias

* [DynamicJasper - Repositorio oficial](https://github.com/intive-FDV/DynamicJasper?tab=readme-ov-file)
