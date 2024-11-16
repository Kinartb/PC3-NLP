# PC3 - NLP: Integración de subword embeddings en redes neuronales recurrentes (RNNs) para procesamiento de lenguaje natural
Curso: CC0C2A Procesamiento del Lenguaje Natural 
Alumno: Arturo Hinostroza Olivera

Archivos subidos:

- **BPE.iypbn**: Ejemplo básico de embeddings con BPE
- **binario**: contiene los archivos 
  - **subword.model**: Contiene la configuración interna y las reglas necesarias para realizar la segmentación de texto en subwords.
  - **subowrd.vocab**: Este archivo contiene el vocabulario generado durante el entrenamiento con el dataset.
  - **PC3_binario.py**: Archivo de clasificación binaria con usando **IMDB Dataset of 50K Movie Reviews**, dataset de películas y ver si es una buena o mala película.
  - **binario.txt**: Resultados del código binario

- **multiclase**: contiene los archivos
  - **subwordmulti.model**: Contiene la configuración interna y las reglas necesarias para realizar la segmentación de texto en subwords.
  - **subowrdmulti.vocab**: Este archivo contiene el vocabulario generado durante el entrenamiento con el dataset.
  - **PC3_multiclase.py**: Archivo de clasificación multiclase (5 clases) usando **# Amazon Fine Food Reviews**, dataset de comida y ver su valoración del 1 al 5.
  - **multiclase.txt**: Resultados del código multiclase.

Referencias:
- [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) 
-  [Amazon Fine Food ](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
-  [Subword embedding](https://d2l.ai/chapter_natural-language-processing-pretraining/subword-embedding.html#the-fasttext-model)
-  [SentencePiece](https://github.com/google/sentencepiece)
# Byte Pair Encoding (BPE)


## **1. Concepto**

BPE es un algoritmo de compresión que identifica y fusiona iterativamente los pares de símbolos consecutivos más frecuentes en un conjunto de datos, creando subpalabras de longitud variable. Este método se ha utilizado en modelos de preentrenamiento para manejar eficientemente palabras raras y fuera del vocabulario.

---

## **2. Ejemplo de algoritmo BPE**

Ejemplo de BPE: [BPE.ipynb]([https://example.com](https://github.com/Kinartb/PC3-NLP/blob/main/BPE.ipynb))

# **Código**

```
VOCAB_SIZE = 8000
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_CLASSES = 2  # Clasificación binaria
BATCH_SIZE = 16  
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
MAX_LEN = 128  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

1.  **`VOCAB_SIZE` (8000):** Tamaño del vocabulario que se usará en el modelo de subwords.
2.  **`EMBEDDING_DIM` (100):** Dimensiones de los vectores de embedding que representarán las palabras o subwords.
3.  **`HIDDEN_DIM` (128):** Número de unidades en la capa oculta de la RNN.
4.  **`NUM_CLASSES` (2):** Clasificación binaria para el sentimiento (`0` = negativo, `1` = positivo).
5.  **`BATCH_SIZE` (16):** Número de muestras procesadas simultáneamente durante el entrenamiento.
6.  **`NUM_EPOCHS` (10):** Iteraciones completas sobre todo el dataset.
7.  **`LEARNING_RATE` (0.001):** Tasa de aprendizaje para el optimizador.
8.  **`MAX_LEN` (128):** Longitud máxima de las secuencias procesadas (truncadas si exceden este valor).
9.  **`device`:** Automáticamente selecciona GPU si está disponible; de lo contrario, usa CPU.

```
def load_imdb_dataset():
    import kagglehub
    path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    data_path = f"{path}/IMDB Dataset.csv"
    df = pd.read_csv(data_path)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    data = list(zip(df['review'], df['sentiment']))
    return data

``` 
-   **Descarga del Dataset:** Utiliza `kagglehub` para descargar un dataset de reseñas de películas IMDB con sus etiquetas de sentimiento.
-   **Preprocesamiento:**
    -   Convierte las etiquetas categóricas (`positive`, `negative`) en valores numéricos (`1`, `0`).
    -   Devuelve una lista de tuplas donde cada entrada contiene una reseña y su etiqueta.
``` 
def train_sentencepiece(data, model_prefix="subword"):
    text_file = "data.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        for text, _ in data:
            f.write(text + "\n")
    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=model_prefix,  
        vocab_size=VOCAB_SIZE,
        character_coverage=0.9995,
        model_type='bpe'
    )
    os.remove(text_file)
 ``` 


Este bloque entrena un modelo de SentencePiece para generar subwords usando BPE.

1.  **Preparación de Datos:**
    -   Crea un archivo temporal (`data.txt`) con todas las reseñas de texto del dataset.
2.  **Entrenamiento del Modelo:**
    -   Usa `SentencePieceTrainer.train` para generar un modelo de subwords (`subword.model`) basado en las oraciones de entrada.
    -   `vocab_size`: Define el número máximo de subwords (8000 en este caso).
    -   `model_type`: Especifica el uso del algoritmo BPE.


 ``` 
class SubwordDataset(Dataset):
    def __init__(self, data, sp):
        self.data = []
        self.sp = sp
        for text, label in data:
            tokens = self.sp.encode(text, out_type=int)[:MAX_LEN]
            if len(tokens) > 0:
                self.data.append((tokens, label))

 ``` 


Este dataset convierte texto en tokens usando el modelo de subwords (`SentencePieceProcessor`).

-   **Tokenización:** Cada texto se convierte en una secuencia de índices de subwords, truncada a `MAX_LEN`.
-   **Almacenamiento:** Guarda las secuencias tokenizadas junto con sus etiquetas.

 ``` 
 class TraditionalDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = []
        self.vocab = vocab
        self.unk_idx = self.vocab.get("<unk>")
        for text, label in data:
            tokens = [self.vocab.get(word, self.unk_idx) for word in text.split()][:MAX_LEN]
            if len(tokens) > 0:
                self.data.append((tokens, label))
  ``` 

-   **Vocabulario:** Asigna un índice único a cada palabra.
-   **Tokenización:** Convierte palabras en índices; usa `<unk>` para palabras fuera del vocabulario.

  ``` 
def build_vocab(data, max_vocab_size=VOCAB_SIZE):
    word_freq = Counter()
    for text, _ in data:
        word_freq.update(text.split())
    most_common = word_freq.most_common(max_vocab_size - 2)
    vocab = {word: idx+2 for idx, (word, _) in enumerate(most_common)}
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1
    return vocab
  ``` 

Crea un vocabulario a partir de palabras en el dataset.
-   **Frecuencias:** Cuenta la aparición de cada palabra.
-   **Índices Especiales:** `<pad>` (0) para padding y `<unk>` (1) para palabras desconocidas.


  ``` 
  class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout=0.5):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
  ``` 



-   **Embedding:** Convierte índices en vectores densos de dimensión `EMBEDDING_DIM`.
-   **GRU:** Procesa secuencias y produce una representación compacta en la última celda oculta.
-   **Dropout:** Evita el sobreajuste.
-   **Clasificador:** Una capa lineal para predecir la clase (`positivo` o `negativo`)


  ``` 
def train_and_evaluate(model, train_loader, val_loader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            texts, labels, lengths = batch
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
  ``` 
  
Entrena el modelo calculando la pérdida y ajustando los pesos en cada iteración. Al final de cada época, evalúa el desempeño en el conjunto de validación.

Ejecución del modelo:
  ``` 
if __name__ == '__main__':
    DATA = load_imdb_dataset()
    train_sentencepiece(DATA)
    sp = spm.SentencePieceProcessor()
    sp.load("subword.model")
    vocab = build_vocab(DATA)
    model_subword = RNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)
    model_traditional =
  ``` 
# Resultados:

# Resultados de los Modelos RNN: Subword vs Tradicional  en [binario](https://github.com/Kinartb/PC3-NLP/blob/main/binario/binario.txt)

A continuación, se analizan los resultados obtenidos para dos modelos RNN entrenados para clasificación de texto: uno basado en subword embeddings (**Modelo Subword**) y otro que utiliza una tokenización tradicional (**Modelo Tradicional**).

---

## **Modelo Subword**

### **Evolución de la Pérdida y Precisión**
- **Pérdida de Entrenamiento**: 
  - Disminuye desde **0.6787** (época 1) hasta **0.2888** (época 10), indicando que el modelo aprende correctamente a minimizar el error.
- **Precisión de Entrenamiento**: 
  - Mejora gradualmente de **55.24%** a **87.60%** a lo largo de las 10 épocas.
- **Pérdida y Precisión de Validación**: 
  - Pérdida inicial de **0.5380** y precisión de **73.06%**.
  - La precisión máxima es **84.36%** (época 9). Sin embargo, se observa un ligero aumento en la pérdida hacia el final, sugiriendo posible **sobreajuste**.

### **Tiempo de Ejecución**
- **183.47 segundos**. Es significativamente más rápido que el modelo Tradicional.

---

## **Modelo Tradicional**

### **Evolución de la Pérdida y Precisión**
- **Pérdida de Entrenamiento**: 
  - Disminuye de **0.6665** (época 1) a **0.2731** (época 10), mostrando una convergencia adecuada.
- **Precisión de Entrenamiento**: 
  - Mejora de **57.44%** a **88.42%**, mostrando una excelente capacidad de aprendizaje.
- **Pérdida y Precisión de Validación**:
  - La precisión máxima es **85.18%** (época 6), pero la pérdida aumenta en épocas posteriores, lo que indica **sobreajuste**.

### **Tiempo de Ejecución**
- **285.57 segundos**. Es considerablemente más lento que el modelo Subword.

---

## **Comparación entre Modelos**

| **Aspecto**                | **Modelo Subword**                 | **Modelo Tradicional**             |
|-----------------------------|------------------------------------|------------------------------------|
| **Precisión Máxima Validación** | 84.36%                          | 85.18%                            |
| **Tendencia al Sobreajuste**   | Leve hacia el final              | Más pronunciada hacia el final     |
| **Velocidad de Ejecución**     | Más rápido (**183.47 segundos**) | Más lento (**285.57 segundos**)    |
| **Eficiencia del Vocabulario** | Alta (subwords reutilizables)    | Baja (vocabulario completo)        |

---


# Resultados de los Modelos RNN: Subword vs Tradicional [multiclase](https://github.com/Kinartb/PC3-NLP/blob/main/multiclase/multiclase.txt)

En este análisis, se comparan dos modelos RNN: uno basado en subword embeddings (**Modelo Subword**) y otro con tokenización tradicional (**Modelo Tradicional**). Los resultados se evalúan en términos de pérdida, precisión y tiempo de ejecución.

---

## **Modelo Subword**

### **Evolución de la Pérdida y Precisión**
- **Pérdida de Entrenamiento**: 
  - Inicial: **0.8153** (época 1)
  - Final: **0.6731** (época 10)
  - Disminuye progresivamente, indicando que el modelo aprende correctamente.
- **Precisión de Entrenamiento**: 
  - Mejora de **69.90%** (época 1) a **74.51%** (época 10).
- **Pérdida y Precisión de Validación**: 
  - Pérdida inicial: **0.6981**, con precisión de **73.79%**.
  - Mejor precisión: **75.53%** (época 6), aunque la pérdida final muestra un ligero aumento.

### **Tiempo de Ejecución**
- **3522.56 segundos**. Entrenamiento significativamente más largo.

---

## **Modelo Tradicional**

### **Evolución de la Pérdida y Precisión**
- **Pérdida de Entrenamiento**: 
  - Inicial: **0.8311** (época 1)
  - Final: **0.6727** (época 10).
  - Disminuye, aunque ligeramente más lenta que en el modelo Subword.
- **Precisión de Entrenamiento**: 
  - Mejora de **69.40%** (época 1) a **74.59%** (época 10).
- **Pérdida y Precisión de Validación**: 
  - Pérdida inicial: **0.7166**, con precisión de **73.14%**.
  - Mejor precisión: **75.55%** (época 10).

### **Tiempo de Ejecución**
- **3973.36 segundos**, un tiempo considerablemente mayor al del modelo Subword.

---

## **Comparación entre Modelos**

| **Aspecto**                | **Modelo Subword**                 | **Modelo Tradicional**             |
|-----------------------------|------------------------------------|------------------------------------|
| **Precisión Máxima Validación** | 75.53%                          | 75.55%                            |
| **Tendencia al Sobreajuste**   | Mínima                          | Muy ligera                        |
| **Velocidad de Ejecución**     | Más rápido (**3522.56 segundos**) | Más lento (**3973.36 segundos**)  |
| **Eficiencia del Vocabulario** | Alta (subwords reutilizables)    | Baja (vocabulario completo)        |

---

## **Conclusiones**

El **Modelo Subword** es ideal para entrenamientos rápidos y vocabularios dinámicos, mientras que el **Modelo Tradicional** ofrece una precisión ligeramente superior a costa de mayor tiempo y menos eficiencia.



