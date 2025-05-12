# Client REPO:

https://github.com/Romanow33/rag-client

[![Ver demo](ruta-a-thumbnail.jpg)]( https://drive.google.com/file/d/1QMOcws2mze155BwLHuWQoi7yqx76QcxZ/view?usp=sharing)
 


# 🧠 AI - OpenRouter - RAG Server

Servidor simple de recuperación aumentada con generación (RAG) utilizando [OpenRouter](https://openrouter.ai/) y [Qdrant](https://qdrant.tech/).

---

## 🚀 Instalación y Ejecución

1. Crear entorno virtual:
   ```bash
   python3 -m venv venv
   ```

2. Activar el entorno:
   ```bash
   source venv/bin/activate
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Iniciar el servidor:
   ```bash
   uvicorn info:app --reload
   ```

---

## ⚙️ Variables de Entorno

Configura las siguientes variables de entorno antes de ejecutar el servidor:

| Variable               | Descripción |
|------------------------|-------------|
| `QDRANT_URL`           | URL del servidor Qdrant (local o en la nube). |
| `QDRANT_API_KEY`       | Clave API de Qdrant (si es requerida). |
| `QDRANT_COLLECTION`    | Nombre de la colección en Qdrant. No es necesario crearla previamente; el servidor la generará automáticamente. |
| `EMBEDDING_MODEL`      | Modelo de embeddings de HuggingFace, por ejemplo: `sentence-transformers/paraphrase-MiniLM-L6-v2`. |
| `OPENROUTER_MODEL`     | Modelo para generar respuestas. Puedes usar modelos gratuitos de OpenRouter, como: `deepseek/deepseek-chat-v3-0324:free`. |
| `OPENROUTER_BASE_URL`  | URL base de la API de OpenRouter. |
| `OPENROUTER_API_KEY`   | Clave API de OpenRouter. Puedes obtenerla registrándote en [openrouter.ai](https://openrouter.ai). |

---

## 📦 Ejemplos de Uso

### 1. Subir un PDF para Ingesta

Envía un archivo PDF para que sea procesado e indexado automáticamente en Qdrant:

```bash
curl -X POST http://localhost:8000/upload_pdf \
  -F 'file=@/ruta/a/tu/archivo.pdf'
```

Respuesta esperada:

```json
{
  "message": "'archivo.pdf' recibido. Ingesta en Qdrant programada."
}
```

### 2. Realizar una Consulta

Envía una pregunta al sistema, que buscará en los documentos cargados y generará una respuesta:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Qué trata el archivo PDF que subí?"
  }'
```

Respuesta esperada:

```json
{
  "question": "¿Qué trata el archivo PDF que subí?",
  "answer": "El archivo trata sobre..."
}
```

---

## 📝 Notas

- Asegúrate de tener un servidor Qdrant activo antes de usar este backend.
- Compatible con modelos de HuggingFace para embeddings y OpenRouter para generación de texto.

---

## 🧩 Licencia

MIT License. Sientete libre de modificar y utilizar este proyecto.

---
