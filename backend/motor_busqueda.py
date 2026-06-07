import pandas as pd
import numpy as np
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
stemmer = PorterStemmer()

class MotorUPScholar:
    def __init__(self, ruta_csv):
        self.df = pd.read_csv(ruta_csv).fillna("")
        self.ids = self.df['Paper_Id'].tolist()
        self.abstracts = self.df['Abstract'].tolist()
        self.titulos = self.df['Title'].tolist()
        self.keywords = self.df['Keywords'].tolist()
        self.sesiones = self.df['Session'].tolist()
        
        self.abstracts_proc = [self._steamming(self._normalizacion(doc)) for doc in self.abstracts]
        self.titulos_proc = [self._steamming(self._normalizacion(doc)) for doc in self.titulos]
        self.keywords_proc = [self._steamming(self._normalizacion(doc)) for doc in self.keywords]
        
        self.matriz_abstracts = self._coseno_tfidf(self._construir_tfidf(self.abstracts_proc))
        vocab_titulos = list(set([palabra for doc in self.titulos_proc for palabra in doc]))
        self.matriz_titulos = self._matriz_similitud_jaccard(self.titulos_proc, vocab_titulos)
        vocab_keywords = list(set([palabra for doc in self.keywords_proc for palabra in doc]))
        self.matriz_keywords = self._matriz_similitud_jaccard(self.keywords_proc, vocab_keywords)
        
        self.matriz_final_tradicional = (self.matriz_titulos * 0.1) + (self.matriz_keywords * 0.2) + (self.matriz_abstracts * 0.7)

        self.modelo_llm = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings_abstracts = self.modelo_llm.encode(self.abstracts)

    def obtener_paper_por_id(self, paper_id):
        resultado = self.df[self.df['Paper_Id'] == int(paper_id)]
        if not resultado.empty:
            return resultado.iloc[0].to_dict()
        return None

    def _normalizacion(self, texto):
        doc = re.sub(r'[^a-zA-Z]', ' ', str(texto)).lower().split()
        return [w for w in doc if w not in stopwords.words('english')]

    def _steamming(self, doc_tokens):
        return [stemmer.stem(token) for token in doc_tokens]

    def _construir_tfidf(self, lista_documentos_procesados):
        N = len(lista_documentos_procesados)
        frecuencia_documentos = Counter()
        for doc in lista_documentos_procesados:
            for palabra in set(doc):
                frecuencia_documentos[palabra] += 1
        vocabulario = list(frecuencia_documentos.keys())
        m = []
        for doc in lista_documentos_procesados:
            frecuencias = Counter(doc)
            m.append([frecuencias.get(palabra, 0) for palabra in vocabulario])
        tf = pd.DataFrame(m).T
        with np.errstate(divide='ignore'):
            tf_pesado = np.where(tf > 0, 1 + np.log10(tf), 0)
        idf = np.array([np.log10(N / (frecuencia_documentos[palabra] if frecuencia_documentos[palabra] > 0 else 1)) for palabra in vocabulario])
        matriz_tfidf = tf_pesado * idf.reshape(-1, 1)
        return pd.DataFrame(matriz_tfidf, index=vocabulario)

    def _coseno_tfidf(self, matriz_tfidf):
        normas = np.linalg.norm(matriz_tfidf, axis=0)
        normas = np.where(normas == 0, 1, normas)
        matriz_norm = matriz_tfidf / normas
        return matriz_norm.T.dot(matriz_norm)

    def _matriz_similitud_jaccard(self, lista_documentos_procesados, vocabulario_stem):
        m = [[doc.count(palabra) for doc in lista_documentos_procesados] for palabra in vocabulario_stem]
        df_jaccard = pd.DataFrame(m)
        num_docs = df_jaccard.shape[1]
        matriz_similitud = np.zeros((num_docs, num_docs))
        for i in range(num_docs):
            for j in range(num_docs):
                set_A = set(df_jaccard.index[df_jaccard[i] > 0])
                set_B = set(df_jaccard.index[df_jaccard[j] > 0])
                interseccion = set_A.intersection(set_B)
                union = set_A.union(set_B)
                if len(union) > 0:
                    matriz_similitud[i, j] = len(interseccion) / len(union)
        return pd.DataFrame(matriz_similitud)

    def buscar_tradicional(self, consulta_usuario, filtro_sesion=None):
        consulta_proc = self._steamming(self._normalizacion(consulta_usuario))
        textos_busqueda = self.abstracts_proc + [consulta_proc]
        matriz_busqueda = self._coseno_tfidf(self._construir_tfidf(textos_busqueda))
        
        indice_consulta = len(textos_busqueda) - 1
        similitudes_consulta = matriz_busqueda.iloc[indice_consulta, :-1].copy()
        
        # Filtro Global
        if filtro_sesion:
            for i, session in enumerate(self.sesiones):
                if session != filtro_sesion:
                    similitudes_consulta.iloc[i] = -1
                    
        top_10_indices = similitudes_consulta.nlargest(10).index.tolist()
        top_10_indices = [idx for idx in top_10_indices if similitudes_consulta.iloc[idx] > -1]
        
        resultados = []
        for idx in top_10_indices:
            similitudes_doc = self.matriz_final_tradicional.iloc[idx].copy()
            
            # Candado para que las recomendaciones NO salgan de la sesión elegida
            if filtro_sesion:
                for i, session in enumerate(self.sesiones):
                    if session != filtro_sesion:
                        similitudes_doc.iloc[i] = -1
                        
            for excluido in top_10_indices:
                similitudes_doc.iloc[excluido] = -1 
                
            top_3_indices = similitudes_doc.nlargest(3).index.tolist()
            top_3_indices = [r for r in top_3_indices if similitudes_doc.iloc[r] > -1]
            recomendaciones = [{"id": int(self.ids[r]), "titulo": self.titulos[r], "similitud": round(float(similitudes_doc.iloc[r]), 4)} for r in top_3_indices]
            
            resultados.append({
                "id": int(self.ids[idx]),
                "titulo": self.titulos[idx],
                "abstract": self.abstracts[idx],
                "similitud": round(float(similitudes_consulta.iloc[idx]), 4),
                "recomendaciones": recomendaciones
            })
        return resultados

    def buscar_llm(self, consulta_usuario, filtro_sesion=None):
        embedding_consulta = self.modelo_llm.encode([consulta_usuario])
        similitudes_busqueda = cosine_similarity(embedding_consulta, self.embeddings_abstracts)[0].copy()
        
        # Filtro Global
        if filtro_sesion:
            for i, session in enumerate(self.sesiones):
                if session != filtro_sesion:
                    similitudes_busqueda[i] = -1
                    
        top_10_indices = similitudes_busqueda.argsort()[::-1][:10]
        top_10_indices = [idx for idx in top_10_indices if similitudes_busqueda[idx] > -1]
        
        resultados = []
        for idx in top_10_indices:
            similitudes_doc = cosine_similarity([self.embeddings_abstracts[idx]], self.embeddings_abstracts)[0].copy()
            
            # Candado para que las recomendaciones NO salgan de la sesión elegida
            if filtro_sesion:
                for i, session in enumerate(self.sesiones):
                    if session != filtro_sesion:
                        similitudes_doc[i] = -1
                        
            for i in top_10_indices:
                similitudes_doc[i] = -1
                
            top_3_recomendados = similitudes_doc.argsort()[::-1][:3]
            top_3_recomendados = [r for r in top_3_recomendados if similitudes_doc[r] > -1]
            recomendaciones = [{"id": int(self.ids[r]), "titulo": self.titulos[r], "similitud": round(float(similitudes_doc[r]), 4)} for r in top_3_recomendados]
            
            resultados.append({
                "id": int(self.ids[idx]),
                "titulo": self.titulos[idx],
                "abstract": self.abstracts[idx],
                "similitud": round(float(similitudes_busqueda[idx]), 4),
                "recomendaciones": recomendaciones
            })
        return resultados